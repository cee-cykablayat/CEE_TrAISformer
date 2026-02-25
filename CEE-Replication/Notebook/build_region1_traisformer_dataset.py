#!/usr/bin/env python3
"""Build TrAISformer-compatible train/valid/test pickle files from region_1 parquet."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("CEE-Replication/Notebook/region_1_interpolated.parquet"),
        help="Input parquet with Time, MMSI, Latitude, Longitude, SOG, COG columns.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("CEE-Replication/traisformer_data/region_1"),
        help="Output directory for region_1_{train,valid,test}.pkl and metadata.json.",
    )
    parser.add_argument(
        "--max-seqlen-plus-one",
        type=int,
        default=121,
        help="Window length used by original script (max_seqlen + 1).",
    )
    parser.add_argument(
        "--min-seqlen",
        type=int,
        default=37,
        help="Minimum window length to keep (min_seqlen + 1 from config).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=60,
        help="Sliding window stride.",
    )
    parser.add_argument(
        "--max-sog",
        type=float,
        default=30.0,
        help="SOG normalization cap to match original config behavior.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic MMSI split.",
    )
    return parser.parse_args()


def normalize_clip(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    denom = max(hi - lo, 1e-12)
    out = (values - lo) / denom
    return np.clip(out, 0.0, 0.9999).astype(np.float64)


def split_mmsi(mmsis: np.ndarray, seed: int) -> tuple[set[int], set[int], set[int]]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(mmsis, copy=True)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    train = set(map(int, shuffled[:n_train]))
    valid = set(map(int, shuffled[n_train : n_train + n_valid]))
    test = set(map(int, shuffled[n_train + n_valid :]))
    return train, valid, test


def build_samples(
    df: pd.DataFrame,
    max_len: int,
    min_len: int,
    stride: int,
) -> list[dict]:
    samples: list[dict] = []
    grouped = df.groupby("MMSI", sort=False)
    for mmsi, g in tqdm(grouped, desc="Building vessel windows"):
        g = g.sort_values("Time")
        arr = g[["lat_n", "lon_n", "sog_n", "cog_n", "ts_unix"]].to_numpy(dtype=np.float64)
        n = len(arr)
        if n < min_len:
            continue

        starts = range(0, max(n - min_len + 1, 1), stride)
        emitted = 0
        for st in starts:
            ed = min(st + max_len, n)
            chunk = arr[st:ed]
            if len(chunk) < min_len:
                continue
            mmsi_col = np.full((len(chunk), 1), float(mmsi), dtype=np.float64)
            traj = np.concatenate([chunk, mmsi_col], axis=1)
            samples.append({"mmsi": int(mmsi), "traj": traj})
            emitted += 1

        if emitted == 0 and n >= min_len:
            mmsi_col = np.full((n, 1), float(mmsi), dtype=np.float64)
            traj = np.concatenate([arr, mmsi_col], axis=1)
            samples.append({"mmsi": int(mmsi), "traj": traj})
    return samples


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cols = ["Time", "MMSI", "Latitude", "Longitude", "SOG", "COG"]
    df = pd.read_parquet(args.input, columns=cols)
    df = df.dropna(subset=cols).copy()
    df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
    df = df.dropna(subset=["Time"])
    df["ts_unix"] = (df["Time"].astype("int64") // 10**9).astype(np.float64)
    df["MMSI"] = df["MMSI"].astype(np.int64)

    lat_min = float(df["Latitude"].min())
    lat_max = float(df["Latitude"].max())
    lon_min = float(df["Longitude"].min())
    lon_max = float(df["Longitude"].max())

    df["lat_n"] = normalize_clip(df["Latitude"].to_numpy(np.float64), lat_min, lat_max)
    df["lon_n"] = normalize_clip(df["Longitude"].to_numpy(np.float64), lon_min, lon_max)
    sog = np.clip(df["SOG"].to_numpy(np.float64), 0.0, args.max_sog)
    df["sog_n"] = np.clip(sog / max(args.max_sog, 1e-12), 0.0, 0.9999)
    cog = np.mod(df["COG"].to_numpy(np.float64), 360.0)
    df["cog_n"] = np.clip(cog / 360.0, 0.0, 0.9999)

    df = df.sort_values(["MMSI", "Time"])
    samples = build_samples(
        df=df,
        max_len=args.max_seqlen_plus_one,
        min_len=args.min_seqlen,
        stride=args.stride,
    )

    unique_mmsis = np.array(sorted({int(s["mmsi"]) for s in samples}), dtype=np.int64)
    train_mmsi, valid_mmsi, test_mmsi = split_mmsi(unique_mmsis, args.seed)

    train = [s for s in samples if s["mmsi"] in train_mmsi]
    valid = [s for s in samples if s["mmsi"] in valid_mmsi]
    test = [s for s in samples if s["mmsi"] in test_mmsi]

    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        out = args.outdir / f"region_1_{name}.pkl"
        with out.open("wb") as f:
            pickle.dump(data, f)
        print(f"Saved {name}: {len(data)} samples -> {out}")

    metadata = {
        "input_parquet": str(args.input),
        "n_rows": int(len(df)),
        "n_windows_total": int(len(samples)),
        "n_mmsi": int(len(unique_mmsis)),
        "splits": {
            "train_windows": int(len(train)),
            "valid_windows": int(len(valid)),
            "test_windows": int(len(test)),
            "train_mmsi": int(len(train_mmsi)),
            "valid_mmsi": int(len(valid_mmsi)),
            "test_mmsi": int(len(test_mmsi)),
        },
        "normalization": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "max_sog": float(args.max_sog),
            "cog_divisor": 360.0,
            "clip_max": 0.9999,
        },
        "windowing": {
            "max_seqlen_plus_one": int(args.max_seqlen_plus_one),
            "min_seqlen": int(args.min_seqlen),
            "stride": int(args.stride),
        },
        "suggested_model_sizes": {
            "lat_size": 250,
            "lon_size": 270,
            "sog_size": 30,
            "cog_size": 72,
        },
    }
    meta_out = args.outdir / "metadata.json"
    meta_out.write_text(json.dumps(metadata, indent=2))
    print(f"Saved metadata -> {meta_out}")


if __name__ == "__main__":
    main()
