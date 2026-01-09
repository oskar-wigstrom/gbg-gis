#!/usr/bin/env python3
"""Download Sweden-wide PBF to data/sweden-latest.osm.pbf (idempotent)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import shutil
import urllib.request
import logging
import yaml
import subprocess
import time
import pickle
import gzip
import networkx as nx
import osmnx as ox
from sqlalchemy import create_engine, text
from gbg_gis.download_map import load_config, validate_network_type, process_pbf
from gbg_gis.load_from_file import get_db_settings, make_conn_str
from gbg_gis.routing import NetworkXRouter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PBF_PATH = DATA_DIR / "sweden-latest.osm.pbf"
URL = "https://download.geofabrik.de/europe/sweden-latest.osm.pbf"
MAPS_PATH = PROJECT_ROOT / "config" / "maps.yaml"
SHORTEST_PATH = PROJECT_ROOT / "config" / "shortest.yaml"
DEFAULT_DB_CONFIG = PROJECT_ROOT / "config" / "db.yaml"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def download_file(url: str, dest: Path, force_refresh: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force_refresh:
        logger.info(f"PBF already exists at {dest}; skipping download")
        return
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    logger.info(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(tmp_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    tmp_path.replace(dest)
    logger.info(f"Saved PBF to {dest}")


def download_pbf(force_refresh: bool = False) -> None:
    download_file(URL, PBF_PATH, force_refresh)


def load_maps(path: Path = MAPS_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("maps", [])


def load_shortest_config(path: Path = SHORTEST_PATH) -> list[dict]:
    """Load shortest path query configurations."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("queries", [])


def ensure_metadata_table(engine) -> None:
    ddl = """
        CREATE TABLE IF NOT EXISTS ingested_maps (
            place TEXT NOT NULL,
            network_type TEXT NOT NULL,
            gpkg_path TEXT,
            loaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(place, network_type)
        );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def ensure_osmium() -> str | None:
    osmium_cmd = shutil.which("osmium")
    if not osmium_cmd and Path("/usr/bin/osmium").exists():
        osmium_cmd = "/usr/bin/osmium"
    if not osmium_cmd:
        logger.warning("osmium-tool not found; filtering will be skipped")
        return None
    return osmium_cmd


def filter_with_osmium(osmium_cmd: str, input_pbf: Path, output_pbf: Path, bbox: list[float]) -> None:
    bbox_str = ",".join(map(str, bbox))
    logger.info(f"Filtering {input_pbf} -> {output_pbf} with bbox {bbox_str}")
    output_pbf.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        osmium_cmd,
        "extract",
        "-b",
        bbox_str,
        str(input_pbf),
        "-o",
        str(output_pbf),
        "--overwrite",
    ], check=True)


def maybe_download_base(force: bool) -> None:
    if PBF_PATH.exists() and not force:
        logger.info(f"Base PBF already exists at {PBF_PATH}; skipping download")
        return
    download_pbf(force_refresh=True)


def load_distance_file(file_path: Path) -> dict:
    """Load distance data from either YAML or compressed pickle format."""
    if file_path.suffix == '.gz' and file_path.stem.endswith('.pkl'):
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.suffix == '.yaml':
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def generate_distances_for_query(
    query_config: dict,
    pbf_path: Path,
    cache_dir: Path,
    network_type: str = "all"
) -> dict:
    """Generate shortest path distances for a single query configuration."""
    name = query_config.get("name", "unnamed")
    lat = query_config.get("lat")
    lon = query_config.get("lon")
    address = query_config.get("address")

    # Resolve coordinates
    if lat is not None and lon is not None:
        logger.info(f"Using coordinates for '{name}': {lat:.6f}, {lon:.6f}")
    elif address:
        logger.info(f"Geocoding address for '{name}': {address}")
        location = ox.geocode(address)
        if isinstance(location, (list, tuple)):
            lat, lon = location
        else:
            lat, lon = location.y, location.x
        logger.info(f"Geocoded to: {lat:.6f}, {lon:.6f}")
    else:
        raise ValueError(f"Query '{name}' must have either lat/lon or address")

    # Load graph
    logger.info(f"Loading road network for '{name}'...")
    router = NetworkXRouter(str(pbf_path), cache_dir)
    G = router._load_graph(network_type)
    logger.info(f"Loaded graph with {len(G):,} nodes and {G.number_of_edges():,} edges")

    # Use largest component
    if not nx.is_weakly_connected(G):
        logger.info("Finding largest connected component...")
        largest = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        logger.info(f"Using largest component: {len(G):,} nodes")

    if len(G) <= 1:
        raise RuntimeError(f"Graph has only {len(G)} node(s)")

    # Find nearest node
    target_node = router._nearest_nodes(G, lon, lat)
    logger.info(f"Target node for '{name}': {target_node}")

    # Compute distances
    logger.info(f"Computing shortest path distances for '{name}'...")
    undirected = G.to_undirected(as_view=False)
    lengths = nx.single_source_dijkstra_path_length(undirected, target_node, weight="length")
    logger.info(f"Computed distances for {len(lengths):,} nodes")

    # Build result
    node_ids = []
    distance_ms = []
    lons = []
    lats = []

    for n, d in tqdm(lengths.items(), desc=f"Processing '{name}'", unit="nodes"):
        node = G.nodes[n]
        lon_val = node.get('x', node.get('lon'))
        lat_val = node.get('y', node.get('lat'))
        node_ids.append(int(n))
        distance_ms.append(float(d))
        lons.append(float(lon_val))
        lats.append(float(lat_val))

    return {
        "meta": {
            "query": name,
            "db_column": query_config.get("db_column", f"dist_{name}"),
            "geocoded_lon": lon,
            "geocoded_lat": lat,
            "source_pbf": str(pbf_path),
            "network_type": network_type,
            "node_count": len(node_ids),
            "target_node": target_node,
        },
        "distances": {
            "node_id": node_ids,
            "distance_m": distance_ms,
            "lon": lons,
            "lat": lats,
        },
    }


def load_or_generate_distances(
    shortest_config: list[dict],
    shortest_dir: Path,
    pbf_path: Path,
    cache_dir: Path,
    network_type: str = "all"
) -> dict | None:
    """Load existing distance files or generate missing ones, combine in-memory."""
    if not shortest_config:
        return None

    shortest_dir.mkdir(parents=True, exist_ok=True)

    headers = []
    all_distances = []

    for query_config in shortest_config:
        name = query_config.get("name", "unnamed")
        pkl_path = shortest_dir / f"{name}.pkl.gz"

        if pkl_path.exists():
            logger.info(f"Loading existing distances for '{name}' from {pkl_path}")
            with gzip.open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        else:
            logger.info(f"Generating distances for '{name}' (file not found: {pkl_path})")
            data = generate_distances_for_query(
                query_config, pbf_path, cache_dir, network_type
            )

            logger.info(f"Saving distances to {pkl_path}")
            with gzip.open(pkl_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            file_size_mb = pkl_path.stat().st_size / (1024 * 1024)
            logger.info(f"Saved {pkl_path} ({file_size_mb:.1f} MB)")

        headers.append(data.get("meta", {}))

        dist_data = data.get("distances", {})
        node_ids = dist_data.get("node_id", [])
        distances = dist_data.get("distance_m", [])
        all_distances.append(dict(zip(node_ids, distances)))

    # Combine in-memory
    logger.info(f"Combining {len(headers)} distance queries in-memory...")

    all_node_ids = set()
    for dist_map in all_distances:
        all_node_ids.update(dist_map.keys())

    logger.info(f"Total unique nodes: {len(all_node_ids):,}")

    distances_map = {}
    for nid in tqdm(all_node_ids, desc="Combining distances", unit="nodes"):
        distances_map[nid] = [
            dist_map.get(nid) for dist_map in all_distances
        ]

    return {
        'map': distances_map,
        'headers': headers
    }


def process_maps(
    force_download: bool,
    force_filter: bool,
    upload: bool,
    db_config: Path,
    chunksize: int,
    force_update: bool,
    maps_path: Path,
    shortest_path: Path | None = None,
    cache_dir: Path = PROJECT_ROOT / "cache"
) -> None:
    maps = load_maps(maps_path)
    if not maps:
        logger.error("No maps found in config/maps.yaml")
        sys.exit(1)

    osmium_cmd = ensure_osmium()

    engine = None
    if upload:
        pg_settings, _ = load_config(db_config)
        conn = make_conn_str(get_db_settings(pg_settings))
        engine = create_engine(conn)
        ensure_metadata_table(engine)

    # Load shortest path config and get/generate distances
    distances_data = None
    shortest_config = load_shortest_config(shortest_path or SHORTEST_PATH)

    if shortest_config:
        logger.info(f"Found {len(shortest_config)} shortest path queries in config")
        shortest_dir = PROJECT_ROOT / "data" / "shortest"

        # Determine PBF to use for routing (use filtered if available)
        routing_pbf = DATA_DIR / "sweden-latest.osm_filtered.osm.pbf"
        if not routing_pbf.exists():
            routing_pbf = PBF_PATH

        distances_data = load_or_generate_distances(
            shortest_config,
            shortest_dir,
            routing_pbf,
            cache_dir,
            network_type="all"
        )

        if distances_data:
            logger.info(f"Loaded {len(distances_data['map'])} distance arrays with {len(distances_data['headers'])} queries")
    else:
        logger.info("No shortest path config found, skipping distance computation")

    total = len(maps)
    for idx, entry in enumerate(maps, start=1):
        logger.info(f"[Map {idx}/{total}] Starting")
        url = entry.get("url") or URL
        pbf_rel = entry.get("pbf_path")
        bbox = entry.get("bounding_box") or entry.get("bbox")
        network_type = validate_network_type(entry.get("network_type", "driving"))
        if not pbf_rel:
            filename = Path(url).name or "download.osm.pbf"
            pbf_path = DATA_DIR / filename
        else:
            pbf_path = (PROJECT_ROOT / pbf_rel) if not Path(pbf_rel).is_absolute() else Path(pbf_rel)

        logger.info(f"[Map {idx}/{total}] Download/check {pbf_path}")
        download_file(url, pbf_path, force_refresh=force_download)

        filtered_path = None
        if bbox:
            filtered_path = pbf_path.with_name(f"{pbf_path.stem}_filtered.osm.pbf")
            if filtered_path.exists() and not force_filter:
                logger.info(f"[Map {idx}/{total}] Filtered PBF already exists; skipping: {filtered_path}")
            elif osmium_cmd:
                logger.info(f"[Map {idx}/{total}] Filtering with osmium")
                filter_with_osmium(osmium_cmd, pbf_path, filtered_path, bbox)
            else:
                logger.warning("osmium not available; cannot filter; skipping")

        if upload and engine:
            chosen_pbf = filtered_path if filtered_path and filtered_path.exists() else pbf_path
            logger.info(f"[Map {idx}/{total}] Uploading to database: {chosen_pbf}")
            logger.info(f"[Map {idx}/{total}] Network type: {network_type}, Bounding box: {bbox}, Force update: {force_update}")
            process_pbf(
                engine,
                chosen_pbf,
                network_type=network_type,
                chunksize=chunksize,
                distances_data=distances_data,
            )
        logger.info(f"[Map {idx}/{total}] Done")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and process OSM PBF files based on YAML configuration")
    parser.add_argument("--maps", type=Path, default=MAPS_PATH, help="Path to maps.yaml")
    parser.add_argument("--shortest", type=Path, default=SHORTEST_PATH, help="Path to shortest.yaml config")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "cache", help="Directory for cached graphs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_maps(
        force_download=False,
        force_filter=False,
        upload=True,
        db_config=DEFAULT_DB_CONFIG,
        chunksize=5000,
        force_update=False,
        maps_path=args.maps,
        shortest_path=args.shortest,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
