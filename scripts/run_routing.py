# shortest_path_dump.py
"""
Compute shortest path distances from a query point to all nodes in a network.

Query point: lon=12.0116, lat=57.6952
Source PBF: data/sweden-latest.osm_filtered.osm.pbf
Network type: driving
"""

import argparse
import pickle
import gzip
import re
from datetime import datetime
from pathlib import Path

import networkx as nx
import osmnx as ox
from tqdm import tqdm
from gbg_gis.routing import NetworkXRouter

QUERY_LON = 12.0116
QUERY_LAT = 57.6952
PBF_PATH = "data/sweden-latest.osm_filtered.osm.pbf"
OUTPUT_DIR = Path("data/shortest")


def slugify(name: str) -> str:
    # Allow Swedish characters while collapsing other characters to dashes
    cleaned = re.sub(r"[^a-z0-9åäö]+", "-", name.strip().lower())
    return cleaned.strip("-") or "query"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute distances from a geocoded street to all nodes and save compressed data")
    parser.add_argument("street", nargs='?', help="Street address or place name to geocode")
    parser.add_argument("--pbf", default=PBF_PATH, help="Path to the filtered OSM PBF file")
    parser.add_argument("--network-type", default="all", help="Network type for routing (default: all)")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory to write compressed outputs")
    parser.add_argument("--cache-dir", default="cache", help="Directory for cached graphs")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild the cached graph for this network type")
    # New coordinate-based API
    parser.add_argument("--lat", type=float, help="Latitude coordinate (alternative to geocoding)")
    parser.add_argument("--lon", type=float, help="Longitude coordinate (alternative to geocoding)")
    parser.add_argument("--name", help="Save name for the output file (required when using coordinates)")
    return parser.parse_args()


def main():
    args = parse_args()
    query = args.street
    pbf_path = args.pbf
    network_type = args.network_type
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


    if args.lat is not None and args.lon is not None:
        # Use provided coordinates
        lat, lon = args.lat, args.lon
        if not args.name:
            raise ValueError("Save name must be specified with --name when using coordinates")
        query = args.name  # Use the provided name
        print(f"Using provided coordinates: {lat:.6f}, {lon:.6f}")
        print(f"Save name: {query}")
    else:
        # Use geocoding
        if not query:
            raise ValueError("Either provide a street address or use --lat/--lon with --name")
        print(f"Geocoding: {query}")
        location = ox.geocode(query)
        if isinstance(location, (list, tuple)):
            lat, lon = location
        else:
            lat, lon = location.y, location.x
        print(f"Geocoded to: {lat:.6f}, {lon:.6f}")

    print("Loading road network...")
    router = NetworkXRouter(pbf_path, cache_dir)
    if args.rebuild_cache:
        cache_file = router._cache_file(network_type)
        if cache_file.exists():
            print(f"Removing cached graph: {cache_file}")
            cache_file.unlink()

    G = router._load_graph(network_type)
    print(f"Loaded graph with {len(G):,} nodes and {G.number_of_edges():,} edges")

    # Stay on the largest component to avoid single-node runs
    if not nx.is_weakly_connected(G):
        print("Graph not weakly connected, finding largest component...")
        largest = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest).copy()
        print(f"Using largest component: {len(G):,} nodes")

    if len(G) <= 1:
        raise RuntimeError(f"Loaded graph has only {len(G)} node(s); try rebuilding cache or checking the PBF.")

    print("Finding nearest node to query point...")
    target_node = router._nearest_nodes(G, lon, lat)
    print(f"Target node: {target_node}")

    # Compute distances from all nodes to target using undirected view to avoid one-way isolation
    print("Computing shortest path distances...")
    undirected = G.to_undirected(as_view=False)

    # Add progress bar for distance computation
    print("Running Dijkstra's algorithm...")
    lengths = nx.single_source_dijkstra_path_length(undirected, target_node, weight="length")

    if len(lengths) <= 1:
        raise RuntimeError(
            "Distances computed only for a single node; the graph may be disconnected or caching may be stale. "
            "Use --rebuild-cache or check the PBF/network_type."
        )

    print(f"Computed distances for {len(lengths):,} nodes")
    print("Preparing output data...")

    node_ids: list[int] = []
    distance_ms: list[float] = []
    lons: list[float] = []
    lats: list[float] = []

    # Add progress bar for data preparation
    for n, d in tqdm(lengths.items(), desc="Processing nodes", unit="nodes"):
        node = G.nodes[n]
        lon_val = node.get('x', node.get('lon'))
        lat_val = node.get('y', node.get('lat'))
        node_ids.append(int(n))
        distance_ms.append(float(d))
        lons.append(float(lon_val))
        lats.append(float(lat_val))

    slug = slugify(query)
    # Use compressed pickle instead of YAML
    out_path = output_dir / f"{slug}.pkl.gz"

    data = {
        "meta": {
            "query": query,
            "geocoded_lon": lon,
            "geocoded_lat": lat,
            "source_pbf": pbf_path,
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

    print(f"Writing compressed data to {out_path}...")
    with gzip.open(out_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Show file size comparison
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {len(node_ids):,} nodes to {out_path}")
    print(f"File size: {file_size_mb:.1f} MB (compressed)")


if __name__ == "__main__":
    main()
