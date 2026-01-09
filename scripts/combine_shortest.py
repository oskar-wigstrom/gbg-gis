#!/usr/bin/env python3
"""
Combine all shortest-path files in data/shortest/ into a single file.
Supports both YAML (.yaml) and compressed pickle (.pkl.gz) formats.
Header: sequence of metadata (one per input file).
Nodes: for each node_id, a sequence of distances corresponding to the header order.
"""
import pickle
import gzip
import yaml
from pathlib import Path
import sys
from tqdm import tqdm

SRC_DIR = Path("data/shortest")
OUT_PATH_PKL = Path("data/distances.pkl.gz")

def load_distance_file(file_path: Path) -> dict:
    """Load distance data from either YAML or compressed pickle format."""
    if file_path.suffix == '.gz' and file_path.stem.endswith('.pkl'):
        # Compressed pickle format
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)
    elif file_path.suffix == '.yaml':
        # YAML format
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def main():
    # Find all distance files (both formats)
    yaml_files = list(SRC_DIR.glob("*.yaml"))
    pickle_files = list(SRC_DIR.glob("*.pkl.gz"))
    all_files = sorted(yaml_files + pickle_files)

    if not all_files:
        print(f"No distance files found in {SRC_DIR}")
        sys.exit(1)

    print(f"Found {len(all_files)} distance files:")
    for f in all_files:
        print(f"  {f.name}")

    headers = []
    node_distances = {}
    node_lons = {}
    node_lats = {}

    for idx, file_path in enumerate(tqdm(all_files, desc="Loading files")):
        print(f"Loading {file_path.name}...")
        try:
            data = load_distance_file(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

        meta = data.get("meta", {})
        headers.append(meta)
        dist = data.get("distances", {})
        ids = dist.get("node_id", [])
        dists = dist.get("distance_m", [])
        lons = dist.get("lon", [])
        lats = dist.get("lat", [])

        print(f"  Processing {len(ids):,} nodes...")
        for i, nid in enumerate(tqdm(ids, desc=f"  File {idx+1}", leave=False, unit="nodes")):
            if nid not in node_distances:
                node_distances[nid] = [None] * len(all_files)
                node_lons[nid] = lons[i] if i < len(lons) else None
                node_lats[nid] = lats[i] if i < len(lats) else None
            node_distances[nid][idx] = dists[i] if i < len(dists) else None

    print(f"Building combined output with {len(node_distances):,} unique nodes...")

    # Build output structure
    out = {
        "header": headers,
        "nodes": [
            {
                "node_id": nid,
                "lon": node_lons[nid],
                "lat": node_lats[nid],
                "distances": node_distances[nid],
            }
            for nid in tqdm(sorted(node_distances), desc="Building output", unit="nodes")
        ],
    }

    # Write compressed pickle format only
    print(f"Writing compressed pickle to {OUT_PATH_PKL}...")
    with gzip.open(OUT_PATH_PKL, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Show file size
    pkl_size_mb = OUT_PATH_PKL.stat().st_size / (1024 * 1024)

    print(f"\nOutput file created:")
    print(f"  Compressed pickle: {OUT_PATH_PKL} ({pkl_size_mb:.1f} MB)")
    print(f"Total unique nodes: {len(node_distances):,}")
    print(f"Distance queries: {len(headers)}")

if __name__ == "__main__":
    main()

