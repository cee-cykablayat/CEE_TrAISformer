#!/usr/bin/env python3
"""Modular preprocessing + MLflow training utilities for TrAISformer replication."""

from __future__ import annotations

import json
import math
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


def haversine_nm(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized haversine distance in nautical miles."""
    r_km = 6371.0088
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    km = r_km * c
    return km / 1.852


@dataclass
class PreprocessingConfig:
    max_gap_hours: float = 2.0
    min_voyage_messages: int = 20
    min_voyage_hours: float = 4.0
    max_empirical_speed_knots: float = 40.0
    max_sog_knots: float = 30.0
    downsample_minutes: int = 10
    max_seq_hours: float = 20.0
    keep_only_underway: bool = True
    coastline_min_nm: float = 1.0


class AISPreprocessor:
    """Implements paper-style preprocessing in modular steps."""

    def __init__(self, cfg: PreprocessingConfig):
        self.cfg = cfg

    def load(self, parquet_path: Path) -> pd.DataFrame:
        cols = [
            "Time",
            "MMSI",
            "Latitude",
            "Longitude",
            "SOG",
            "COG",
            "Navstatus",
            "is_interpolated",
        ]
        df = pd.read_parquet(parquet_path)
        cols_present = [c for c in cols if c in df.columns]
        df = df[cols_present].copy()
        df["Time"] = pd.to_datetime(df["Time"], utc=True, errors="coerce")
        df = df.dropna(subset=["Time", "MMSI", "Latitude", "Longitude", "SOG", "COG"]).copy()
        df["MMSI"] = df["MMSI"].astype(np.int64)
        return df.sort_values(["MMSI", "Time"])

    def remove_near_coastline(self, df: pd.DataFrame, coastline_pickle: Optional[Path]) -> pd.DataFrame:
        """Optional coastline filtering if shoreline polygons are available + geopandas/shapely installed."""
        if coastline_pickle is None or not coastline_pickle.exists():
            return df
        try:
            import shapely.geometry as sgeom
            import shapely.ops as sops
        except Exception:
            print("shapely not installed; skipping coastline distance filter.")
            return df

        with open(coastline_pickle, "rb") as f:
            polys = pickle.load(f)
        if not polys:
            return df
        merged = sops.unary_union(polys)
        # Approximation: 1 degree latitude ~ 60 nautical miles.
        max_deg = self.cfg.coastline_min_nm / 60.0
        keep = []
        for lat, lon in zip(df["Latitude"].values, df["Longitude"].values):
            pt = sgeom.Point(float(lon), float(lat))
            keep.append(pt.distance(merged) > max_deg)
        return df.loc[np.array(keep, dtype=bool)].copy()

    def remove_unrealistic_and_stationary(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[df["SOG"] < self.cfg.max_sog_knots].copy()
        if self.cfg.keep_only_underway:
            if "Navstatus" in out.columns:
                nav = out["Navstatus"].astype(str).str.lower()
                bad_nav = nav.str.contains("moored") | nav.str.contains("anchor")
                out = out.loc[~bad_nav].copy()
            out = out[out["SOG"] > 0.1].copy()
        return out

    def split_contiguous(self, df: pd.DataFrame) -> pd.DataFrame:
        max_gap = pd.Timedelta(hours=self.cfg.max_gap_hours)
        dfs: List[pd.DataFrame] = []
        for mmsi, g in tqdm(df.groupby("MMSI", sort=False), desc="Split contiguous"):
            g = g.sort_values("Time").copy()
            gaps = g["Time"].diff() > max_gap
            seg_id = gaps.fillna(False).cumsum().astype(np.int64)
            g["voyage_id"] = g["MMSI"].astype(str) + "_" + seg_id.astype(str)
            dfs.append(g)
        return pd.concat(dfs, ignore_index=True)

    def remove_abnormal_empirical_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        keep_mask = np.ones(len(df), dtype=bool)
        for _, g in tqdm(df.groupby("voyage_id", sort=False), desc="Empirical speed filter"):
            idx = g.index.to_numpy()
            if len(g) < 2:
                keep_mask[idx] = False
                continue
            lat = g["Latitude"].to_numpy(np.float64)
            lon = g["Longitude"].to_numpy(np.float64)
            t = g["Time"].astype("int64").to_numpy(np.float64) / 1e9
            d_nm = haversine_nm(lat[:-1], lon[:-1], lat[1:], lon[1:])
            dt_h = np.maximum((t[1:] - t[:-1]) / 3600.0, 1e-9)
            v_knots = d_nm / dt_h
            ok = np.ones(len(g), dtype=bool)
            ok[1:] = v_knots <= self.cfg.max_empirical_speed_knots
            keep_mask[idx] = ok
        return df.loc[keep_mask].copy()

    def downsample(self, df: pd.DataFrame) -> pd.DataFrame:
        freq = f"{self.cfg.downsample_minutes}min"
        out = []
        for vid, g in tqdm(df.groupby("voyage_id", sort=False), desc="Downsample"):
            g = g.sort_values("Time").set_index("Time")
            # nearest sample per 10-min bucket
            rs = g.resample(freq).first().dropna(subset=["MMSI", "Latitude", "Longitude", "SOG", "COG"])
            if len(rs) == 0:
                continue
            rs = rs.reset_index()
            rs["voyage_id"] = vid
            out.append(rs)
        if not out:
            return df.iloc[0:0].copy()
        return pd.concat(out, ignore_index=True)

    def filter_short_voyages(self, df: pd.DataFrame) -> pd.DataFrame:
        keep_vid = []
        min_dur = pd.Timedelta(hours=self.cfg.min_voyage_hours)
        for vid, g in df.groupby("voyage_id", sort=False):
            if len(g) < self.cfg.min_voyage_messages:
                continue
            dur = g["Time"].max() - g["Time"].min()
            if dur < min_dur:
                continue
            keep_vid.append(vid)
        return df[df["voyage_id"].isin(keep_vid)].copy()

    def split_long_voyages(self, df: pd.DataFrame) -> pd.DataFrame:
        max_rows = int((self.cfg.max_seq_hours * 60) // self.cfg.downsample_minutes)
        chunks = []
        for vid, g in tqdm(df.groupby("voyage_id", sort=False), desc="Split long voyages"):
            g = g.sort_values("Time")
            for i in range(0, len(g), max_rows):
                c = g.iloc[i : i + max_rows].copy()
                if len(c) >= self.cfg.min_voyage_messages:
                    c["voyage_id"] = f"{vid}_chunk{i//max_rows}"
                    chunks.append(c)
        if not chunks:
            return df.iloc[0:0].copy()
        return pd.concat(chunks, ignore_index=True)

    def run(self, parquet_path: Path, coastline_pickle: Optional[Path] = None) -> pd.DataFrame:
        df = self.load(parquet_path)
        print("Loaded:", len(df))
        df = self.remove_near_coastline(df, coastline_pickle)
        print("After coastline filter:", len(df))
        df = self.remove_unrealistic_and_stationary(df)
        print("After SOG/nav filter:", len(df))
        df = self.split_contiguous(df)
        print("After contiguous split:", len(df), "rows,", df["voyage_id"].nunique(), "voyages")
        df = self.remove_abnormal_empirical_speed(df)
        print("After empirical speed filter:", len(df))
        df = self.downsample(df)
        print("After downsample:", len(df))
        df = self.filter_short_voyages(df)
        print("After short voyage filter:", len(df), "rows,", df["voyage_id"].nunique(), "voyages")
        df = self.split_long_voyages(df)
        print("After long voyage split:", len(df), "rows,", df["voyage_id"].nunique(), "voyages")
        return df


@dataclass
class ExportConfig:
    max_sog: float = 30.0
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    seed: int = 42


class TrAISformerDataExporter:
    """Export voyage dataframe into TrAISformer pkl format."""

    def __init__(self, cfg: ExportConfig):
        self.cfg = cfg

    def _normalize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        out = df.copy()
        lat_min = float(out["Latitude"].min())
        lat_max = float(out["Latitude"].max())
        lon_min = float(out["Longitude"].min())
        lon_max = float(out["Longitude"].max())
        out["lat_n"] = np.clip((out["Latitude"] - lat_min) / max(lat_max - lat_min, 1e-12), 0.0, 0.9999)
        out["lon_n"] = np.clip((out["Longitude"] - lon_min) / max(lon_max - lon_min, 1e-12), 0.0, 0.9999)
        out["sog_n"] = np.clip(out["SOG"] / max(self.cfg.max_sog, 1e-12), 0.0, 0.9999)
        out["cog_n"] = np.clip(np.mod(out["COG"], 360.0) / 360.0, 0.0, 0.9999)
        out["ts_unix"] = (out["Time"].astype("int64") // 10**9).astype(np.float64)
        meta = {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max,
            "max_sog": float(self.cfg.max_sog),
        }
        return out, meta

    def _to_samples(self, df: pd.DataFrame) -> List[Dict]:
        samples = []
        for _, g in tqdm(df.groupby("voyage_id", sort=False), desc="Build samples"):
            arr = g[["lat_n", "lon_n", "sog_n", "cog_n", "ts_unix"]].to_numpy(np.float64)
            mmsi = int(g["MMSI"].iloc[0])
            mmsi_col = np.full((len(arr), 1), float(mmsi), dtype=np.float64)
            traj = np.concatenate([arr, mmsi_col], axis=1)
            samples.append({"mmsi": mmsi, "traj": traj})
        return samples

    def _split(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        rng = np.random.default_rng(self.cfg.seed)
        mmsi = np.array(sorted({int(s["mmsi"]) for s in samples}), dtype=np.int64)
        rng.shuffle(mmsi)
        n = len(mmsi)
        n_train = int(n * self.cfg.train_ratio)
        n_valid = int(n * self.cfg.valid_ratio)
        tr = set(map(int, mmsi[:n_train]))
        va = set(map(int, mmsi[n_train : n_train + n_valid]))
        te = set(map(int, mmsi[n_train + n_valid :]))
        train = [s for s in samples if s["mmsi"] in tr]
        valid = [s for s in samples if s["mmsi"] in va]
        test = [s for s in samples if s["mmsi"] in te]
        return train, valid, test

    def export(self, df: pd.DataFrame, out_dir: Path) -> Dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        norm_df, norm_meta = self._normalize(df)
        samples = self._to_samples(norm_df)
        train, valid, test = self._split(samples)
        for name, data in [("train", train), ("valid", valid), ("test", test)]:
            with open(out_dir / f"region_1_{name}.pkl", "wb") as f:
                pickle.dump(data, f)
        meta = {
            "normalization": norm_meta,
            "splits": {
                "train_windows": len(train),
                "valid_windows": len(valid),
                "test_windows": len(test),
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta


@dataclass
class EarlyStoppingConfig:
    max_epochs: int = 50
    patience: int = 8
    min_delta: float = 1e-4


class CheckpointManager:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.last_ckpt = self.run_dir / "last_checkpoint.pt"
        self.best_ckpt = self.run_dir / "best_model.pt"
        self.config_path = self.run_dir / "training_config.json"
        self.history_path = self.run_dir / "history.json"

    def save(self, payload: Dict, is_best: bool) -> None:
        torch.save(payload, self.last_ckpt)
        if is_best:
            torch.save(payload, self.best_ckpt)

    def load(self) -> Optional[Dict]:
        if self.last_ckpt.exists():
            return torch.load(self.last_ckpt, map_location="cpu")
        return None


class TrAISformerMLflowTrainer:
    """Crash-resumable trainer with MLflow logging."""

    def __init__(
        self,
        model,
        train_dataset,
        valid_dataset,
        test_dataset,
        config,
        sample_fn,
        run_dir: Path,
        early_stop: EarlyStoppingConfig,
        experiment_name: str = "trAISformer_region1",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.sample_fn = sample_fn
        self.device = config.device
        self.early_stop = early_stop
        self.ckpt = CheckpointManager(run_dir)
        self.experiment_name = experiment_name
        self.history: Dict[str, List[float]] = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}
        self.start_epoch = 0
        self.best_valid = float("inf")
        self.best_epoch = -1
        self.patience_count = 0
        self.optimizer = self.model.configure_optimizers(config)
        self._try_resume()

    def _try_resume(self):
        payload = self.ckpt.load()
        if payload is None:
            return
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optim_state"])
        self.start_epoch = int(payload["epoch"]) + 1
        self.best_valid = float(payload["best_valid"])
        self.best_epoch = int(payload["best_epoch"])
        self.patience_count = int(payload["patience_count"])
        self.history = payload.get("history", self.history)
        print(f"Resumed from epoch {payload['epoch']} with best_valid={self.best_valid:.6f}")

    @staticmethod
    def _batch_accuracy(logits, targets):
        lat_size, lon_size, sog_size, cog_size = targets["sizes"]
        lat_logits, lon_logits, sog_logits, cog_logits = torch.split(logits, (lat_size, lon_size, sog_size, cog_size), dim=-1)
        lat_pred = lat_logits.argmax(dim=-1)
        lon_pred = lon_logits.argmax(dim=-1)
        sog_pred = sog_logits.argmax(dim=-1)
        cog_pred = cog_logits.argmax(dim=-1)
        lat_acc = (lat_pred == targets["lat"]).float().mean()
        lon_acc = (lon_pred == targets["lon"]).float().mean()
        sog_acc = (sog_pred == targets["sog"]).float().mean()
        cog_acc = (cog_pred == targets["cog"]).float().mean()
        return (lat_acc + lon_acc + sog_acc + cog_acc) / 4.0

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        losses = []
        accs = []
        pbar = tqdm(loader, total=len(loader), disable=False)
        for seqs, masks, _, _, _ in pbar:
            seqs = seqs.to(self.device)
            masks = masks[:, :-1].to(self.device)
            with torch.set_grad_enabled(train):
                logits, loss = self.model(seqs, masks=masks, with_targets=True)
                loss = loss.mean()
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                    self.optimizer.step()
            idxs, _ = self.model.to_indexes(seqs, mode=getattr(self.model, "partition_mode", "uniform"))
            targets = idxs[:, 1:, :]
            acc = self._batch_accuracy(
                logits,
                {
                    "lat": targets[:, :, 0],
                    "lon": targets[:, :, 1],
                    "sog": targets[:, :, 2],
                    "cog": targets[:, :, 3],
                    "sizes": (self.model.lat_size, self.model.lon_size, self.model.sog_size, self.model.cog_size),
                },
            )
            losses.append(float(loss.detach().cpu()))
            accs.append(float(acc.detach().cpu()))
            pbar.set_description(f"{'train' if train else 'valid'} loss={losses[-1]:.5f} acc={accs[-1]:.4f}")
        return float(np.mean(losses)), float(np.mean(accs))

    def _log_curves(self):
        fig = plt.figure(figsize=(8, 4))
        plt.plot(self.history["train_loss"], label="train_loss")
        plt.plot(self.history["valid_loss"], label="valid_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        curve_path = self.ckpt.run_dir / "loss_curve.png"
        fig.savefig(curve_path, dpi=150)
        plt.close(fig)
        if mlflow:
            mlflow.log_artifact(str(curve_path))

    def train(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(self.valid_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)

        if mlflow:
            mlflow.set_experiment(self.experiment_name)

        run_ctx = mlflow.start_run(run_name="trAISformer_region1") if mlflow else None
        try:
            conf_dump = {k: v for k, v in self.config.__dict__.items() if not k.startswith("_")}
            conf_dump["early_stopping"] = asdict(self.early_stop)
            self.ckpt.config_path.write_text(json.dumps(conf_dump, indent=2, default=str))
            if mlflow:
                mlflow.log_artifact(str(self.ckpt.config_path))
                mlflow.log_params(
                    {
                        "max_epochs": self.early_stop.max_epochs,
                        "patience": self.early_stop.patience,
                        "min_delta": self.early_stop.min_delta,
                        "batch_size": self.config.batch_size,
                        "learning_rate": self.config.learning_rate,
                    }
                )

            for epoch in range(self.start_epoch, self.early_stop.max_epochs):
                train_loss, train_acc = self._run_epoch(train_loader, train=True)
                valid_loss, valid_acc = self._run_epoch(valid_loader, train=False)

                self.history["train_loss"].append(train_loss)
                self.history["valid_loss"].append(valid_loss)
                self.history["train_acc"].append(train_acc)
                self.history["valid_acc"].append(valid_acc)

                improved = valid_loss < (self.best_valid - self.early_stop.min_delta)
                if improved:
                    self.best_valid = valid_loss
                    self.best_epoch = epoch
                    self.patience_count = 0
                else:
                    self.patience_count += 1

                payload = {
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid": self.best_valid,
                    "best_epoch": self.best_epoch,
                    "patience_count": self.patience_count,
                    "history": self.history,
                }
                self.ckpt.save(payload, is_best=improved)
                self.ckpt.history_path.write_text(json.dumps(self.history, indent=2))

                if mlflow:
                    mlflow.log_metrics(
                        {
                            "train_loss": train_loss,
                            "valid_loss": valid_loss,
                            "train_acc": train_acc,
                            "valid_acc": valid_acc,
                            "best_valid_loss": self.best_valid,
                        },
                        step=epoch,
                    )

                self._log_curves()
                print(
                    f"epoch={epoch+1}/{self.early_stop.max_epochs} "
                    f"train_loss={train_loss:.5f} valid_loss={valid_loss:.5f} "
                    f"train_acc={train_acc:.4f} valid_acc={valid_acc:.4f} "
                    f"best_valid={self.best_valid:.5f} patience={self.patience_count}/{self.early_stop.patience}"
                )

                if self.patience_count >= self.early_stop.patience:
                    print("Early stopping triggered.")
                    break

            # Final quick sample artifact from test split
            self.model.eval()
            with torch.no_grad():
                seqs, _, _, _, _ = next(iter(test_loader))
                seqs_init = seqs[:4, : self.config.init_seqlen, :].to(self.device)
                preds = self.sample_fn(
                    self.model,
                    seqs_init,
                    steps=12,
                    temperature=1.0,
                    sample=True,
                    sample_mode=self.config.sample_mode,
                    r_vicinity=self.config.r_vicinity,
                    top_k=self.config.top_k,
                )
            p = preds[0].detach().cpu().numpy()
            fig = plt.figure(figsize=(5, 5))
            plt.plot(p[:, 1], p[:, 0], marker="o", ms=2)
            plt.title("Sampled trajectory")
            plt.xlabel("lon_norm")
            plt.ylabel("lat_norm")
            plt.grid(True)
            plt.tight_layout()
            pred_path = self.ckpt.run_dir / "sample_prediction.png"
            fig.savefig(pred_path, dpi=150)
            plt.close(fig)
            if mlflow:
                mlflow.log_artifact(str(pred_path))
                mlflow.log_artifact(str(self.ckpt.best_ckpt))
                mlflow.log_artifact(str(self.ckpt.last_ckpt))
                mlflow.log_artifact(str(self.ckpt.history_path))
        finally:
            if run_ctx is not None:
                mlflow.end_run()
