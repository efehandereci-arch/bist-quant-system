"""Çoklu varlık veri katmanı ve survivorship bias kontrolü.

Survivorship bias: Bugünkü BIST 100 üyelerini geçmişe taşımak, o tarihte
endekste olmayan (sonradan eklenen) ya da endeksten çıkarılan/işlem görmeyen
hisseleri örneklemden sessizce çıkarır; geçmiş performans iyimser görünür.

Kontrol: ``constituents_file`` (CSV: ``date,ticker``) tarihsel endeks üyelik
anlık görüntülerini içerir. Bir hisse t tarihinde, ``date <= t`` olan en son
anlık görüntüde yer alıyorsa üyedir. Olaylar yalnızca üyelik dönemlerinde
üretilir. Delist olmuş hisselerin fiyat verisi ``csv_dir`` altında
bulunuyorsa aynı şekilde kullanılır. Dosya yoksa bu durum rapora açıkça
"survivorship bias kontrol edilmedi" sınırlaması olarak yazılır.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from ..data import load_ohlcv_csv, simulate_ohlcv, validate_ohlcv
from .settings import DataSettings

log = logging.getLogger("meta_labeling.research")

DEFAULT_BIST_TICKERS = ("SASA", "THYAO", "ASELS", "BIMAS", "AKBNK", "GARAN", "EREGL", "TUPRS")


@dataclass
class MarketData:
    ticker: str
    ohlcv: pd.DataFrame
    source: str
    simulated: bool
    notes: list[str]


def simulation_seed(ticker: str, cfg: DataSettings) -> int:
    """Config listesindeki i. hisse -> simulation.seed + i (ilk hisse orijinal yolu üretir);
    listede olmayan hisseler için ada bağlı deterministik tohum."""
    if ticker in cfg.tickers:
        return cfg.simulation.seed + cfg.tickers.index(ticker)
    return cfg.simulation.seed + 1000 + zlib.crc32(ticker.encode()) % 10_000


def load_market(ticker: str, cfg: DataSettings) -> MarketData:
    """Tek bir hisse için OHLCV yükler (kaynak config'den)."""
    notes: list[str] = []
    if cfg.source == "simulated":
        sim = replace(cfg.simulation, seed=simulation_seed(ticker, cfg))
        df = simulate_ohlcv(sim)
        notes.append("SİMÜLE VERİ: fiyat ve hacim sentetiktir; ADV ve market impact bir varsayımdır.")
        simulated = True
    elif cfg.source == "csv":
        path = Path(cfg.csv_dir) / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{ticker} için veri yok: {path}")
        df = load_ohlcv_csv(path)
        simulated = False
    elif cfg.source == "yfinance":
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise ImportError("yfinance kaynağı için: pip install yfinance") from exc
        raw = yf.download(f"{ticker}{cfg.yfinance_suffix}", period=cfg.yfinance_period,
                          interval="1d", auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise ValueError(f"yfinance veri döndürmedi: {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = validate_ohlcv(raw)
        notes.append("yfinance: düzeltilmiş (auto_adjust) fiyatlar; veri kalitesi doğrulanmalı.")
        simulated = False
    else:
        raise ValueError(f"Bilinmeyen veri kaynağı: {cfg.source}")
    if cfg.start:
        df = df.loc[cfg.start:]
    if cfg.end:
        df = df.loc[: cfg.end]
    return MarketData(ticker=ticker, ohlcv=df, source=cfg.source, simulated=simulated, notes=notes)


def validate_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Veri kalitesi kontrolleri; ciddi sorunlarda hata, diğerlerinde uyarı satırı."""
    checks = []

    def add(name: str, ok: bool, detail: str, fatal: bool = False) -> None:
        checks.append({"check": name, "status": "PASS" if ok else ("FAIL" if fatal else "WARN"), "detail": detail})

    add("index_monotonic", df.index.is_monotonic_increasing, "Tarih indeksi sıralı", fatal=True)
    add("index_unique", df.index.is_unique, "Tekrarlanan tarih yok", fatal=True)
    add("no_nan", not df.isna().any().any(), f"NaN sayısı: {int(df.isna().sum().sum())}", fatal=True)
    add("positive_prices", bool((df[["Open", "High", "Low", "Close"]] > 0).all().all()), "Fiyatlar > 0", fatal=True)
    hl_ok = (df["High"] >= df[["Open", "Close"]].max(axis=1) - 1e-9) & (df["Low"] <= df[["Open", "Close"]].min(axis=1) + 1e-9)
    add("ohlc_consistency", bool(hl_ok.all()), f"High/Low tutarsız bar: {int((~hl_ok).sum())}")
    zero_vol = int((df["Volume"] <= 0).sum())
    add("volume_positive", zero_vol == 0, f"Hacmi sıfır bar: {zero_vol}")
    gaps = df.index.to_series().diff().dt.days
    big = int((gaps > 7).sum())
    add("calendar_gaps", big == 0, f"7 günden uzun boşluk: {big}")
    jumps = int((df["Close"].pct_change().abs() > 0.25).sum())
    add("extreme_moves", jumps == 0, f"|getiri| > %25 bar: {jumps} (split/temettü düzeltmesi kontrol edin)")
    add("length", len(df) >= 500, f"Bar sayısı: {len(df)}", fatal=True)
    table = pd.DataFrame(checks)
    fatal = table[table["status"] == "FAIL"]
    if not fatal.empty:
        raise ValueError("Veri doğrulama başarısız:\n" + fatal.to_string(index=False))
    return table


class SurvivorshipControl:
    """Tarihsel endeks üyeliği; dosya yoksa ``available=False``."""

    def __init__(self, path: str | None):
        self.available = bool(path) and Path(path).exists()
        self.snapshots: pd.DataFrame | None = None
        if self.available:
            snap = pd.read_csv(path)
            snap.columns = [c.lower() for c in snap.columns]
            snap["date"] = pd.to_datetime(snap["date"])
            self.snapshots = snap
        elif path:
            log.warning("constituents_file bulunamadı: %s", path)

    def membership(self, ticker: str, index: pd.DatetimeIndex) -> pd.Series | None:
        if not self.available:
            return None
        snap = self.snapshots
        dates = pd.DatetimeIndex(sorted(snap["date"].unique()))
        members = {d: set(snap.loc[snap["date"] == d, "ticker"]) for d in dates}
        pos = dates.searchsorted(index, side="right") - 1
        flags = [pos_i >= 0 and ticker in members[dates[pos_i]] for pos_i in pos]
        return pd.Series(flags, index=index, name="is_member")

    def limitation(self, n_tickers: int) -> str:
        if self.available:
            return "Tarihsel endeks üyeliği uygulandı; olaylar yalnızca üyelik dönemlerinde üretildi."
        if n_tickers > 1:
            return ("SURVIVORSHIP BIAS KONTROL EDİLMEDİ: tarihsel endeks üyeliği / delist verisi yok. "
                    "Bugünkü hisse listesi geçmişe taşınmıştır; sonuçlar iyimser olabilir.")
        return ("Tek hisse: survivorship bias doğrudan uygulanmaz, ancak hissenin bugün hâlâ işlem görmesi "
                "nedeniyle seçilmiş olması bir seçim yanlılığıdır (ex-post seçim).")
