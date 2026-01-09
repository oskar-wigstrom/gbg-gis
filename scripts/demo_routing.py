#!/usr/bin/env python
"""Minimal demo: route using local PBF (no DB, no geocoding).

Prereqs:
- Ensure you have a local PBF, e.g. data/sweden-latest.osm_filtered.osm.pbf
- The script uses a small set of hardcoded routes within Gothenburg.

Usage:
  uv run python scripts/demo_routing.py --pbf data/sweden-latest.osm_filtered.osm.pbf --network driving
"""
from __future__ import annotations

import argparse
from pathlib import Path

from gbg_gis.routing import NetworkXRouter


def main(pbf: Path, network: str) -> None:
    router = NetworkXRouter(pbf)

    routes = [
        {
            "route_id": 1,
            "name": "Central Station to Liseberg",
            "start_lon": 11.9736,
            "start_lat": 57.7089,
            "end_lon": 12.0116,
            "end_lat": 57.6952,
        },
        {
            "route_id": 2,
            "name": "Nordstan to Slottsskogen",
            "start_lon": 11.9686,
            "start_lat": 57.7086,
            "end_lon": 11.9415,
            "end_lat": 57.6847,
        },
    ]

    print(f"Routing {len(routes)} paths using {network} on {pbf} ...")
    gdf = router.compute_many_routes(routes, network)

    if gdf.empty:
        print("No routes found")
        return

    # Print summary
    for rid in gdf["route_id"].unique():
        sub = gdf[gdf["route_id"] == rid]
        total_m = sub["cumulative_length"].iloc[-1]
        print(f"Route {rid}: {total_m/1000:.2f} km, {len(sub)} segments")

    # Save to GeoPackage for inspection
    out = Path("cache/routes_demo.gpkg")
    out.parent.mkdir(exist_ok=True)
    gdf.to_file(out, layer="routes", driver="GPKG")
    print(f"Saved routes to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Demo routing using local PBF")
    ap.add_argument("--pbf", type=Path, required=True, help="Path to local PBF")
    ap.add_argument("--network", default="driving", choices=["driving", "cycling", "walking", "all", "driving+service"], help="Network type")
    args = ap.parse_args()
    main(args.pbf, args.network)

