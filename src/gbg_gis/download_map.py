"""Load road network from a local OSM PBF and write to GeoPackage cache."""
from __future__ import annotations

import gc
from pathlib import Path

import logging
from pyrosm import OSM
import pandas as pd  # Add at the top with other imports
from sqlalchemy import text

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
    distances_data: dict | None = None,
) -> None:
    """Process the PBF file and load data into the database.

    Args:
        engine: SQLAlchemy engine for database connection
        pbf_path: Path to the PBF file
        network_type: Type of network to extract (driving, walking, cycling, etc.)
        chunksize: Number of rows to insert at once (lower = less memory)
        distances_data: Optional dict with 'map' (node_id -> distances array) and 'headers' (metadata)
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

        # Ensure index is node_id for mapping
        if "id" in nodes.columns and nodes.index.name != "id":
            nodes = nodes.set_index("id")

        # Reset index to make id a regular column for to_postgis
        nodes = nodes.reset_index()

        # Add distance data if provided
        if distances_data is not None:
            distances_map = distances_data.get('map', {})
            headers = distances_data.get('headers', [])

            if distances_map and headers:
                logger.info(f"Adding {len(headers)} distance columns to nodes")

                # Group headers by db_column to handle multiple queries per column
                column_groups = {}
                for i, header in enumerate(headers):
                    col_name = header.get("db_column", f"dist_{i}")
                    if col_name not in column_groups:
                        column_groups[col_name] = []
                    column_groups[col_name].append((i, header))

                # Add distance columns based on grouped db_column names
                for col_name, query_group in column_groups.items():
                    if len(query_group) == 1:
                        # Single query for this column - use scalar values
                        i, header = query_group[0]
                        query_name = header.get("query", f"query_{i}")
                        logger.info(f"Adding scalar column '{col_name}' for query '{query_name}'")

                        def get_distance_for_index(node_id, index=i):
                            distances = distances_map.get(node_id)
                            if distances and index < len(distances) and distances[index] is not None:
                                return float(distances[index])
                            return None

                        nodes[col_name] = nodes['id'].apply(lambda nid: get_distance_for_index(nid, i))
                        filled_count = nodes[col_name].notna().sum()
                        logger.info(f"Scalar column '{col_name}': {filled_count:,} nodes with distances")

                    else:
                        # Multiple queries for this column - create array
                        query_names = [header.get("query", f"query_{i}") for i, header in query_group]
                        logger.info(f"Adding array column '{col_name}' for queries: {', '.join(query_names)}")

                        def get_distances_for_indices(node_id, indices=[i for i, _ in query_group]):
                            distances = distances_map.get(node_id)
                            if distances:
                                result = []
                                for idx in indices:
                                    if idx < len(distances) and distances[idx] is not None:
                                        result.append(float(distances[idx]))
                                    else:
                                        result.append(None)
                                # Only return array if at least one value is not None
                                if any(x is not None for x in result):
                                    # Convert to comma-separated string for PostgreSQL
                                    filtered_result = [str(x) if x is not None else 'NULL' for x in result]
                                    return '{' + ','.join(filtered_result) + '}'
                            return None

                        nodes[col_name] = nodes['id'].apply(lambda nid: get_distances_for_indices(nid))
                        filled_count = nodes[col_name].notna().sum()
                        logger.info(f"Array column '{col_name}': {filled_count:,} nodes with distance arrays")

                # Create metadata table for the headers
                logger.info("Creating distance metadata table")
                with engine.begin() as conn:
                    conn.execute(text("DROP TABLE IF EXISTS distance_metadata"))
                    conn.execute(text("""
                        CREATE TABLE distance_metadata (
                            index_id INTEGER PRIMARY KEY,
                            query TEXT,
                            db_column TEXT,
                            geocoded_lon DOUBLE PRECISION,
                            geocoded_lat DOUBLE PRECISION,
                            source_pbf TEXT,
                            network_type TEXT,
                            node_count INTEGER,
                            target_node BIGINT
                        )
                    """))

                    for i, header in enumerate(headers):
                        conn.execute(text("""
                            INSERT INTO distance_metadata 
                            (index_id, query, db_column, geocoded_lon, geocoded_lat, source_pbf, network_type, node_count, target_node)
                            VALUES (:idx, :query, :col, :lon, :lat, :pbf, :net_type, :count, :target)
                        """), {
                            "idx": i,
                            "query": header.get("query"),
                            "col": header.get("db_column", f"dist_{i}"),
                            "lon": header.get("geocoded_lon"),
                            "lat": header.get("geocoded_lat"),
                            "pbf": header.get("source_pbf"),
                            "net_type": header.get("network_type"),
                            "count": header.get("node_count"),
                            "target": header.get("target_node")
                        })
            else:
                logger.info("No distance data provided")
        else:
            logger.info("No distances data provided")

        # Load nodes in chunks
        logger.info(f"Loading nodes into database (chunksize={chunksize})...")
        with engine.begin() as conn:
            nodes.to_postgis("road_nodes", conn, if_exists="replace", chunksize=chunksize, index=False)

            # Convert array columns to proper PostgreSQL array types
            if distances_data and distances_data.get('headers'):
                headers = distances_data.get('headers', [])

                # Group headers by db_column to identify array columns
                column_groups = {}
                for i, header in enumerate(headers):
                    col_name = header.get("db_column", f"dist_{i}")
                    if col_name not in column_groups:
                        column_groups[col_name] = []
                    column_groups[col_name].append((i, header))

                # Convert array columns to PostgreSQL arrays
                for col_name, query_group in column_groups.items():
                    if len(query_group) > 1:
                        logger.info(f"Converting nodes column '{col_name}' to PostgreSQL array")

                        # Add array column with proper type
                        conn.execute(text(f"ALTER TABLE road_nodes ADD COLUMN {col_name}_array DOUBLE PRECISION[]"))

                        # Convert Python lists to PostgreSQL arrays manually
                        # Get all non-null values and convert them
                        result = conn.execute(text(f"SELECT id, {col_name} FROM road_nodes WHERE {col_name} IS NOT NULL"))

                        for row in result:
                            node_id, array_str = row
                            # Parse the string representation back to a list
                            try:
                                # Handle the string representation of Python lists
                                if array_str.startswith('[') and array_str.endswith(']'):
                                    # Remove brackets and split by comma
                                    values = array_str[1:-1].split(', ')
                                    # Convert to PostgreSQL array format
                                    pg_array = '{' + ','.join(values) + '}'
                                    conn.execute(text(f"""
                                        UPDATE road_nodes 
                                        SET {col_name}_array = :array_val
                                        WHERE id = :node_id
                                    """), {"array_val": pg_array, "node_id": node_id})
                            except Exception as e:
                                logger.warning(f"Failed to convert array for node {node_id}: {e}")

                        # Drop original column and rename array column
                        conn.execute(text(f"ALTER TABLE road_nodes DROP COLUMN {col_name}"))
                        conn.execute(text(f"ALTER TABLE road_nodes RENAME COLUMN {col_name}_array TO {col_name}"))


        del nodes
        gc.collect()

        # Project edges to SWEREF99 TM
        logger.info("Projecting edges to SWEREF99 TM (EPSG:3006)")
        edges = edges.to_crs(epsg=3006)

        # Add distance data to edges if available
        if distances_data and distances_data.get('map'):
            distances_map = distances_data.get('map', {})
            headers = distances_data.get('headers', [])

            logger.info("Adding distance data to roads")

            # Group headers by db_column to handle multiple queries per column
            column_groups = {}
            for i, header in enumerate(headers):
                col_name = header.get("db_column", f"dist_{i}")
                if col_name not in column_groups:
                    column_groups[col_name] = []
                column_groups[col_name].append((i, header))

            # Add distance columns for edges, same grouping as nodes
            for col_name, query_group in column_groups.items():
                if len(query_group) == 1:
                    # Single query for this column - use scalar values
                    i, header = query_group[0]
                    query_name = header.get("query", f"query_{i}")
                    logger.info(f"Adding road scalar column '{col_name}' for query '{query_name}'")

                    def calc_avg_distance_for_index(row, index=i):
                        u_distances = distances_map.get(row['u'])
                        v_distances = distances_map.get(row['v'])

                        u_dist = None
                        v_dist = None

                        if u_distances and index < len(u_distances):
                            u_dist = u_distances[index]
                        if v_distances and index < len(v_distances):
                            v_dist = v_distances[index]

                        if u_dist is not None and v_dist is not None:
                            return (float(u_dist) + float(v_dist)) / 2
                        elif u_dist is not None:
                            return float(u_dist)
                        elif v_dist is not None:
                            return float(v_dist)
                        return None

                    edges[col_name] = edges.apply(lambda row: calc_avg_distance_for_index(row, i), axis=1)
                    filled_count = edges[col_name].notna().sum()
                    logger.info(f"Road scalar column '{col_name}': {filled_count:,} edges with distances")

                else:
                    # Multiple queries for this column - create array
                    query_names = [header.get("query", f"query_{i}") for i, header in query_group]
                    logger.info(f"Adding road array column '{col_name}' for queries: {', '.join(query_names)}")

                    def calc_avg_distances_for_indices(row, indices=[i for i, _ in query_group]):
                        u_distances = distances_map.get(row['u'])
                        v_distances = distances_map.get(row['v'])

                        result = []
                        for idx in indices:
                            u_dist = None
                            v_dist = None

                            if u_distances and idx < len(u_distances):
                                u_dist = u_distances[idx]
                            if v_distances and idx < len(v_distances):
                                v_dist = v_distances[idx]

                            if u_dist is not None and v_dist is not None:
                                result.append((float(u_dist) + float(v_dist)) / 2)
                            elif u_dist is not None:
                                result.append(float(u_dist))
                            elif v_dist is not None:
                                result.append(float(v_dist))
                            else:
                                result.append(None)

                        # Only return array if at least one value is not None
                        if any(x is not None for x in result):
                            # Convert to PostgreSQL array format
                            filtered_result = [str(x) if x is not None else 'NULL' for x in result]
                            return '{' + ','.join(filtered_result) + '}'
                        return None

                    edges[col_name] = edges.apply(lambda row: calc_avg_distances_for_indices(row), axis=1)
                    filled_count = edges[col_name].notna().sum()
                    logger.info(f"Road array column '{col_name}': {filled_count:,} edges with distance arrays")


        # Load edges in chunks
        logger.info(f"Loading edges into database (chunksize={chunksize})...")
        with engine.begin() as conn:
            edges.to_postgis("roads", conn, if_exists="replace", chunksize=chunksize)

            # Convert array columns to proper PostgreSQL array types
            if distances_data and distances_data.get('headers'):
                headers = distances_data.get('headers', [])

                # Group headers by db_column to identify array columns
                column_groups = {}
                for i, header in enumerate(headers):
                    col_name = header.get("db_column", f"dist_{i}")
                    if col_name not in column_groups:
                        column_groups[col_name] = []
                    column_groups[col_name].append((i, header))

                # Convert array columns to PostgreSQL arrays
                for col_name, query_group in column_groups.items():
                    if len(query_group) > 1:
                        logger.info(f"Converting roads column '{col_name}' to PostgreSQL array")

                        # Add array column with proper type
                        conn.execute(text(f"ALTER TABLE roads ADD COLUMN {col_name}_array DOUBLE PRECISION[]"))

                        # Convert Python lists to PostgreSQL arrays manually
                        # Get all non-null values and convert them
                        result = conn.execute(text(f"SELECT u, v, {col_name} FROM roads WHERE {col_name} IS NOT NULL LIMIT 1000"))

                        batch_size = 1000
                        offset = 0

                        while True:
                            result = conn.execute(text(f"SELECT u, v, {col_name} FROM roads WHERE {col_name} IS NOT NULL LIMIT {batch_size} OFFSET {offset}"))
                            rows = result.fetchall()

                            if not rows:
                                break

                            for row in rows:
                                u, v, array_str = row
                                try:
                                    # Handle the string representation of Python lists
                                    if array_str.startswith('[') and array_str.endswith(']'):
                                        # Remove brackets and split by comma
                                        values = array_str[1:-1].split(', ')
                                        # Convert to PostgreSQL array format
                                        pg_array = '{' + ','.join(values) + '}'
                                        conn.execute(text(f"""
                                            UPDATE roads 
                                            SET {col_name}_array = :array_val
                                            WHERE u = :u_val AND v = :v_val
                                        """), {"array_val": pg_array, "u_val": u, "v_val": v})
                                except Exception as e:
                                    logger.warning(f"Failed to convert array for road {u}->{v}: {e}")

                            offset += batch_size

                        # Drop original column and rename array column
                        conn.execute(text(f"ALTER TABLE roads DROP COLUMN {col_name}"))
                        conn.execute(text(f"ALTER TABLE roads RENAME COLUMN {col_name}_array TO {col_name}"))


        del edges
        gc.collect()

        logger.info("PBF processing and database load completed successfully")

    finally:
        # No temporary `filtered_output` variable is maintained by this function;
        # cleanup/logging of any filtered PBF should be handled by the caller
        # if needed. Keep finally block as a no-op to ensure resources are freed.
        pass
