"""Load road network from a local OSM PBF and write to GeoPackage cache."""
from __future__ import annotations

import gc
from pathlib import Path

import logging
from pyrosm import OSM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "db.yaml"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_config(config_path: Path | None = None):
    cfg_path = Path(config_path) if config_path else CONFIG_PATH
    with open(cfg_path, "r", encoding="ascii") as f:
        import yaml
        cfg = yaml.safe_load(f) or {}
    pg = cfg.get("postgres", {})
    data_cfg = cfg.get("data", {})
    return pg, data_cfg


def ensure_data_dir(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "area"


def default_output_path(place_query: str, network_type: str, data_cfg: dict) -> tuple[Path, Path]:
    data_dir = PROJECT_ROOT / data_cfg.get("dir", "data")
    place_slug = slugify(place_query)
    net_slug = slugify(network_type)
    filename = f"{place_slug}_{net_slug}.gpkg"
    return data_dir, data_dir / filename


def validate_network_type(network_type: str) -> str:
    valid_types = {"driving", "driving_psv", "walking", "cycling", "all", "driving+service"}
    if network_type not in valid_types:
        logger.warning(f"Invalid network_type '{network_type}' provided. Defaulting to 'driving'.")
        return "driving"
    return network_type


def crop_pbf_to_bbox(pbf_path: Path, bbox: tuple[float, float, float, float]) -> Path:
    """Crop the PBF file to the specified bounding box."""
    logger.info(f"Cropping PBF file {pbf_path} to bounding box {bbox}")
    cropped_path = pbf_path.with_name(f"{pbf_path.stem}_cropped{pbf_path.suffix}")

    osm = OSM(str(pbf_path), bounding_box=bbox)
    edges = osm.get_network(nodes=False)
    nodes = osm.get_network(nodes=True)

    if edges is None or nodes is None or edges.empty or nodes.empty:
        raise RuntimeError("No network data extracted from PBF for the given bounding box")

    logger.info(f"Saving cropped PBF to {cropped_path}")
    nodes.to_file(cropped_path, layer="road_nodes", driver="GPKG")
    edges.to_file(cropped_path, layer="roads", driver="GPKG")

    return cropped_path


def prefilter_pbf_with_osmium(
    input_pbf: Path,
    output_pbf: Path,
    bounding_box: tuple[float, float, float, float],
) -> Path:
    """Pre-filter a large PBF file using osmium-tool to reduce memory usage.

    Args:
        input_pbf: Path to the input PBF file
        output_pbf: Path for the filtered output PBF file
        bounding_box: (minx, miny, maxx, maxy) bounding box

    Returns:
        Path to the filtered PBF file
    """
    import subprocess
    import shutil

    # Check if osmium is available (try multiple locations)
    osmium_cmd = shutil.which("osmium")
    if not osmium_cmd and Path("/usr/bin/osmium").exists():
        osmium_cmd = "/usr/bin/osmium"

    if not osmium_cmd:
        logger.warning("osmium-tool not found. Install it with: sudo apt-get install osmium-tool")
        logger.warning("Proceeding without pre-filtering (may use more memory)")
        return input_pbf

    minx, miny, maxx, maxy = bounding_box
    bbox_str = f"{minx},{miny},{maxx},{maxy}"

    logger.info(f"Pre-filtering PBF with osmium to bbox: {bbox_str}")

    try:
        subprocess.run(
            [osmium_cmd, "extract", "-b", bbox_str, str(input_pbf), "-o", str(output_pbf), "--overwrite"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"Created filtered PBF: {output_pbf}")
        return output_pbf
    except subprocess.CalledProcessError as e:
        logger.error(f"osmium extract failed: {e.stderr}")
        logger.warning("Falling back to unfiltered PBF (may use more memory)")
        return input_pbf


def process_pbf(
    engine,
    pbf_path: Path,
    network_type: str = "driving",
    chunksize: int = 10000,
) -> None:
    """Process the PBF file and load data into the database.

    Args:
        engine: SQLAlchemy engine for database connection
        pbf_path: Path to the PBF file
        network_type: Type of network to extract (driving, walking, cycling, etc.)
        chunksize: Number of rows to insert at once (lower = less memory)
    """
    logger.info(f"Processing PBF file {pbf_path}")

    network_type = validate_network_type(network_type)

    try:
        # Extract nodes and edges together (get_network with nodes=True returns tuple)
        logger.info("Extracting network from PBF...")
        osm = OSM(str(pbf_path))
        result = osm.get_network(network_type=network_type, nodes=True)

        # get_network with nodes=True returns (nodes, edges) tuple
        if isinstance(result, tuple):
            nodes, edges = result
        else:
            # Fallback if it returns just edges
            edges = result
            nodes = None

        del osm
        gc.collect()

        if nodes is None or (hasattr(nodes, 'empty') and nodes.empty):
            raise RuntimeError("No node data extracted from PBF")
        if edges is None or (hasattr(edges, 'empty') and edges.empty):
            raise RuntimeError("No edge data extracted from PBF")

        logger.info(f"Extracted {len(nodes)} nodes and {len(edges)} edges")

        # Project to SWEREF99 TM
        logger.info("Projecting nodes to SWEREF99 TM (EPSG:3006)")
        nodes = nodes.to_crs(epsg=3006)

        # Load nodes in chunks
        logger.info(f"Loading nodes into database (chunksize={chunksize})...")
        with engine.begin() as conn:
            nodes.to_postgis("road_nodes", conn, if_exists="replace", chunksize=chunksize)

        del nodes
        gc.collect()

        # Project edges to SWEREF99 TM
        logger.info("Projecting edges to SWEREF99 TM (EPSG:3006)")
        edges = edges.to_crs(epsg=3006)

        # Load edges in chunks
        logger.info(f"Loading edges into database (chunksize={chunksize})...")
        with engine.begin() as conn:
            edges.to_postgis("roads", conn, if_exists="replace", chunksize=chunksize)

        del edges
        gc.collect()

        logger.info("PBF processing and database load completed successfully")

    finally:
        if filtered_output and filtered_output.exists() and filtered_output != pbf_path:
            logger.info(f"Filtered PBF retained at {filtered_output}")
