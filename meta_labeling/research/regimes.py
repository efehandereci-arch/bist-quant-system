"""Geleceğe bakmayan piyasa rejimi sınıflandırması.

Tüm eşikler, t anına kadar (t dahil değil) gözlenen dağılımın GENİŞLEYEN
(expanding) kantillerinden hesaplanır. Tüm örneklem üzerinden hesaplanan
kantiller (ör. "tüm verinin %33'lük dilimi") geleceğe bakar ve rejim
analizini sızdırır; burada kullanılmaz. ``min_history`` bardan önceki
dönem ``unknown`` olarak işaretlenir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .settings import RegimeSettings

UNKNOWN = "unknown"


def classify_regimes(ohlcv: pd.DataFrame, vol: pd.Series, cfg: RegimeSettings) -> pd.DataFrame:
    """Bar bazında üç rejim sınıflaması: volatilite, trend, piyasa durumu."""
    close = ohlcv["Close"]
    hist = vol.shift(1).expanding(min_periods=cfg.min_history)
    q_lo, q_hi, q_hv = hist.quantile(cfg.vol_low_q), hist.quantile(cfg.vol_high_q), hist.quantile(cfg.high_vol_q)
    known = q_lo.notna() & vol.notna()

    vol_regime = pd.Series(
        np.select([vol < q_lo, vol > q_hi], ["low", "high"], default="medium"), index=close.index
    ).where(known, UNKNOWN)

    sma = close.rolling(cfg.trend_sma, min_periods=cfg.trend_sma).mean()
    slope = sma - sma.shift(cfg.trend_slope_window)
    trend = pd.Series(
        np.select(
            [(close > sma) & (slope > 0), (close < sma) & (slope < 0)], ["bull", "bear"], default="sideways"
        ),
        index=close.index,
    ).where(slope.notna(), UNKNOWN)

    drawdown = close / close.rolling(cfg.drawdown_window, min_periods=1).max() - 1.0
    high_vol = vol > q_hv
    market = pd.Series(
        np.select(
            [high_vol & (drawdown <= cfg.crisis_drawdown), high_vol],
            ["crisis", "high-volatility"],
            default="normal",
        ),
        index=close.index,
    ).where(known, UNKNOWN)

    return pd.DataFrame({"vol_regime": vol_regime, "trend_regime": trend, "market_regime": market})
