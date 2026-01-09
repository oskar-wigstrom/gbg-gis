"""Load cached Gothenburg roads GeoPackages into PostGIS.

Reads DB config from config.yaml (overridable via env vars) and loads
`road_nodes` and `roads` layers into the target PostGIS database.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import yaml
from sqlalchemy import create_engine, inspect

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "db.yaml"
DEFAULT_LAYERS = ["road_nodes", "roads"]


def load_config(config_path: Path | None = None):
    cfg_path = Path(config_path) if config_path else CONFIG_PATH
    with open(cfg_path, "r", encoding="ascii") as f:
        cfg = yaml.safe_load(f) or {}
    pg = cfg.get("postgres", {})
    data_cfg = cfg.get("data", {})
    return pg, data_cfg


def get_db_settings(pg: dict) -> dict:
    return {
        "user": pg.get("user", "gisuser"),
        "password": pg.get("password", "gispass"),
        "database": pg.get("database", "gisdb"),
        "host": pg.get("host", "localhost"),
        "port": int(pg.get("port", 5432)),
    }


def make_conn_str(settings: dict) -> str:
    return (
        f"postgresql://{settings['user']}:{settings['password']}@"
        f"{settings['host']}:{settings['port']}/{settings['database']}"
    )


def table_exists(engine, table_name: str, schema: str | None = None) -> bool:
    inspector = inspect(engine)
    return inspector.has_table(table_name, schema=schema)


def load_layers(
    engine,
    gpkg_path: Path,
    layers: Iterable[str],
    if_exists: str = "append",
) -> None:
    """Load layers into PostGIS with a configurable merge strategy."""
    if if_exists not in {"fail", "replace", "append"}:
        raise ValueError("if_exists must be one of 'fail', 'replace', or 'append'")

    for layer in layers:
        already = table_exists(engine, layer)
        gdf = gpd.read_file(gpkg_path, layer=layer)
        gdf.to_postgis(name=layer, con=engine, if_exists=if_exists, index=False)
        if not already:
            print(f"Loaded {layer} from {gpkg_path} into DB")
        elif if_exists == "append":
            print(f"Appended {layer} from {gpkg_path} into DB")
        elif if_exists == "replace":
            print(f"Replaced {layer} from {gpkg_path} into DB")


def load_layers_if_missing(engine, gpkg_path: Path, layers: Iterable[str]) -> None:
    for layer in layers:
        if table_exists(engine, layer):
            print(f"Skipping {layer}: already exists in DB")
            continue
        gdf = gpd.read_file(gpkg_path, layer=layer)
        gdf.to_postgis(name=layer, con=engine, if_exists="fail", index=False)
        print(f"Loaded {layer} from {gpkg_path} into DB")
