"""Routing utilities for shortest path queries using NetworkX and OSMnx.

Provides Python interface for:
- Loading road networks from database or downloading fresh
- Computing shortest paths entirely in Python (no pgRouting dependency)
- Batch processing of many routes efficiently
- Saving computed routes to database for QGIS visualization
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from pyrosm import OSM
from shapely.geometry import LineString

NetworkType = Literal["driving", "cycling", "walking", "all", "driving+service"]


class NetworkXRouter:
    """Pure Python routing using a local PBF. No database or geocoding."""

    def __init__(self, pbf_path: Path, cache_dir: Path | None = None):
        self.pbf_path = Path(pbf_path)
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.graphs: dict[tuple[str, str], nx.MultiDiGraph] = {}

    def _cache_file(self, network_type: str) -> Path:
        stem = self.pbf_path.stem
        return self.cache_dir / f"{stem}_{network_type}_graph.pkl"

    def _load_graph(self, network_type: NetworkType) -> nx.MultiDiGraph:
        key = (str(self.pbf_path), network_type)
        if key in self.graphs:
            return self.graphs[key]

        cache_file = self._cache_file(network_type)
        if cache_file.exists():
            with cache_file.open("rb") as f:
                G = pickle.load(f)
        else:
            osm = OSM(str(self.pbf_path))
            nodes, edges = osm.get_network(network_type=network_type, nodes=True)
            if nodes is None or edges is None or nodes.empty or edges.empty:
                raise RuntimeError(f"No network data found in {self.pbf_path} for type '{network_type}'")
            G = ox.graph_from_gdfs(nodes, edges, graph_attrs={"crs": nodes.crs})
            with cache_file.open("wb") as f:
                pickle.dump(G, f)
        self.graphs[key] = G
        return G

    def _nearest_nodes(self, G: nx.MultiDiGraph, lon: float, lat: float) -> int:
        return ox.distance.nearest_nodes(G, lon, lat)

    def _path_to_gdf(self, G: nx.MultiDiGraph, path: list[int], route_id: int) -> gpd.GeoDataFrame:
        rows = []
        cumulative = 0.0
        for seq, (u, v) in enumerate(zip(path[:-1], path[1:]), start=1):
            data = G[u][v][0]
            geom = data.get("geometry")
            if geom is None:
                geom = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])] )
            length = float(data.get("length", geom.length))
            cumulative += length
            rows.append({
                "route_id": route_id,
                "seq": seq,
                "length": length,
                "cumulative_length": cumulative,
                "highway": data.get("highway", "unknown"),
                "geometry": geom,
            })
        return gpd.GeoDataFrame(rows, crs=G.graph.get("crs", "EPSG:4326"))

    def route_between_coords(
        self,
        start_lon: float,
        start_lat: float,
        end_lon: float,
        end_lat: float,
        network_type: NetworkType = "driving",
        route_id: int = 1,
    ) -> gpd.GeoDataFrame | None:
        G = self._load_graph(network_type)
        try:
            start_node = self._nearest_nodes(G, start_lon, start_lat)
            end_node = self._nearest_nodes(G, end_lon, end_lat)
            path = nx.shortest_path(G, start_node, end_node, weight="length")
        except Exception:
            return None
        return self._path_to_gdf(G, path, route_id)

    def compute_many_routes(
        self,
        route_requests: list[dict],
        network_type: NetworkType = "driving",
    ) -> gpd.GeoDataFrame:
        routes = []
        for i, req in enumerate(route_requests, start=1):
            rid = req.get("route_id", i)
            gdf = self.route_between_coords(
                req["start_lon"], req["start_lat"], req["end_lon"], req["end_lat"],
                network_type=network_type,
                route_id=rid,
            )
            if gdf is not None and not gdf.empty:
                gdf["route_name"] = req.get("name", f"Route {rid}")
                routes.append(gdf)
        if routes:
            combined = gpd.GeoDataFrame(pd.concat(routes, ignore_index=True))
            if combined.crs and combined.crs.to_string() != "EPSG:3006":
                combined = combined.to_crs("EPSG:3006")
            return combined
        return gpd.GeoDataFrame(
            columns=["route_id", "seq", "length", "cumulative_length", "highway", "route_name", "geometry"],
            crs="EPSG:3006",
        )


# Convenience function

def route_between_coords(
    pbf_path: Path,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    network_type: NetworkType = "driving",
    cache_dir: Path | None = None,
) -> gpd.GeoDataFrame | None:
    router = NetworkXRouter(pbf_path, cache_dir)
    return router.route_between_coords(start_lon, start_lat, end_lon, end_lat, network_type)
