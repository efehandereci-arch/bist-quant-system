"""Deney kaydı (experiment tracking) ve tekrar üretilebilirlik.

Her çalıştırma için ``<output_dir>/experiments/<experiment_id>.json`` yazılır
ve ``index.csv`` dosyasına bir satır eklenir. Kayıt; tam konfigürasyonu,
veri seti özetini (SHA-256), kod sürümünü (git commit), tohumu ve metrikleri
içerir. ``reproduce()`` kayıttaki konfigürasyonla aynı deneyi yeniden çalıştırır.

Kayıt sayısı aynı zamanda denenen konfigürasyon sayısının dürüst bir
göstergesidir; DSR yorumlanırken dikkate alınmalıdır.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def dataset_version(ohlcv: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(ohlcv, index=True).to_numpy()
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


class ExperimentTracker:
    def __init__(self, output_dir: str | Path):
        self.dir = Path(output_dir) / "experiments"

    def log(self, record: dict[str, Any]) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        record = _jsonable(record)
        path = self.dir / f"{record['experiment_id']}.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        index = self.dir / "index.csv"
        row = {
            "experiment_id": record["experiment_id"],
            "timestamp": record["timestamp"],
            "dataset_version": record["dataset_version"],
            "ticker": record.get("ticker"),
            "config_fingerprint": record["config_fingerprint"],
            "oos_meta_sharpe": record["metrics"].get("meta_sharpe"),
            "verdict": record["metrics"].get("verdict"),
        }
        new = not index.exists()
        with index.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                writer.writeheader()
            writer.writerow(row)
        return path

    def n_experiments(self) -> int:
        index = self.dir / "index.csv"
        if not index.exists():
            return 0
        return max(sum(1 for _ in index.open(encoding="utf-8")) - 1, 0)


def build_record(*, cfg, ticker: str, ohlcv: pd.DataFrame, feature_names: list[str],
                 train_period: tuple, test_period: tuple, metrics: dict[str, Any]) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc)
    ds_ver = dataset_version(ohlcv)
    fp = cfg.fingerprint()
    return {
        "experiment_id": f"{stamp:%Y%m%dT%H%M%S}_{ticker}_{fp[:8]}",
        "timestamp": stamp.isoformat(),
        "ticker": ticker,
        "dataset_version": ds_ver,
        "data_source": cfg.data.source,
        "config_fingerprint": fp,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "seed": cfg.experiment.seed,
        "features": feature_names,
        "primary_model": cfg.to_dict()["primary"],
        "meta_model": cfg.to_dict()["meta_model"],
        "barrier_parameters": cfg.to_dict()["barriers"],
        "cusum_parameters": cfg.to_dict()["cusum"],
        "threshold": cfg.meta_model.threshold,
        "cost": cfg.to_dict()["costs"],
        "execution": cfg.to_dict()["execution"],
        "position_sizing": cfg.to_dict()["sizing"],
        "risk": cfg.to_dict()["risk"],
        "train_period": list(train_period),
        "test_period": list(test_period),
        "metrics": metrics,
        "config": cfg.to_dict(),
    }


def reproduce(record_path: str | Path):
    """Kayıttaki konfigürasyonla deneyi yeniden çalıştırır; ``ResearchSession`` döndürür."""
    from .session import ResearchSession
    from .settings import config_from_dict

    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    session = ResearchSession(config_from_dict(record["config"]), ticker=record["ticker"])
    session.run_all()
    new_version = dataset_version(session.ohlcv)
    if new_version != record["dataset_version"]:
        session.log.warning("Veri seti sürümü farklı: %s != %s", new_version, record["dataset_version"])
    return session
