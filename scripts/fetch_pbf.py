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
from sqlalchemy import create_engine, text
from gbg_gis.download_map import load_config, validate_network_type, process_pbf
from gbg_gis.load_from_file import get_db_settings, make_conn_str

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PBF_PATH = DATA_DIR / "sweden-latest.osm.pbf"
URL = "https://download.geofabrik.de/europe/sweden-latest.osm.pbf"
MAPS_PATH = PROJECT_ROOT / "config" / "maps.yaml"
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
    with open(path, "r", encoding="ascii") as f:
        data = yaml.safe_load(f) or {}
    return data.get("maps", [])


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


def process_maps(force_download: bool, force_filter: bool, upload: bool, db_config: Path, chunksize: int, force_update: bool, maps_path: Path) -> None:
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

    for entry in maps:
        url = entry.get("url") or URL
        pbf_rel = entry.get("pbf_path")
        bbox = entry.get("bounding_box") or entry.get("bbox")
        network_type = validate_network_type(entry.get("network_type", "driving"))
        if not pbf_rel:
            # Default dest based on URL filename
            filename = Path(url).name or "download.osm.pbf"
            pbf_path = DATA_DIR / filename
        else:
            pbf_path = (PROJECT_ROOT / pbf_rel) if not Path(pbf_rel).is_absolute() else Path(pbf_rel)

        # Download base PBF if missing / forced
        download_file(url, pbf_path, force_refresh=force_download)

        filtered_path = None
        if bbox:
            filtered_path = pbf_path.with_name(f"{pbf_path.stem}_filtered.osm.pbf")
            if filtered_path.exists() and not force_filter:
                logger.info(f"Filtered PBF already exists; skipping: {filtered_path}")
            elif osmium_cmd:
                filter_with_osmium(osmium_cmd, pbf_path, filtered_path, bbox)
            else:
                logger.warning("osmium not available; cannot filter; skipping")

        if upload and engine:
            chosen_pbf = filtered_path if filtered_path and filtered_path.exists() else pbf_path
            logger.info(f"Uploading PBF to database: {chosen_pbf}")
            logger.info(f"Network type: {network_type}, Bounding box: {bbox}, Force update: {force_update}")
            process_pbf(
                engine,
                chosen_pbf,
                force_update,
                bounding_box=bbox,
                network_type=network_type,
                chunksize=chunksize,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and process OSM PBF files based on YAML configuration")
    parser.add_argument("--maps", type=Path, default=MAPS_PATH, help="Path to maps.yaml")
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
    )


if __name__ == "__main__":
    main()
