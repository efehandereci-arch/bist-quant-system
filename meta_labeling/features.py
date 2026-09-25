"""Meta-model öznitelikleri: yönden bağımsız piyasa rejimi göstergeleri.

Meta-model "birincil sinyal bu koşullarda güvenilir mi?" sorusunu yanıtlar;
dolayısıyla ihtiyaç duyduğu bilgi yön değil, *ortam*dır: trend mi yatay mı,
volatilite genişliyor mu daralıyor mu, getiriler momentum mu yoksa ortalamaya
dönüş mü sergiliyor? Aşağıdaki özniteliklerin tamamı t barının kapanışına
kadar bilinen veriyle hesaplanır (geriye bakan pencereler, ``shift(-k)`` yok).
Bu özellik ``tests/test_features_weights_sizing.py`` içinde kesilmiş veri üzerinde
doğrulanır.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI (0-100). Aşırı alım/satım rejimini ölçer."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(100.0).where(gain.notna())


def average_true_range(ohlcv: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder ATR: gap'leri de içeren bar içi volatilite ölçüsü."""
    high, low, close = ohlcv["High"], ohlcv["Low"], ohlcv["Close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """Kaufman verimlilik oranı: |net hareket| / toplam yol. 1'e yakın = trend, 0'a yakın = gürültü."""
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return net / path.replace(0.0, np.nan)


def build_feature_matrix(ohlcv: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Bar bazında rejim öznitelik matrisi.

    Öznitelikler:
        rsi            : RSI (aşırı alım/satım)
        atr_pct        : ATR / kapanış (normalize bar içi volatilite)
        atr_vol_ratio  : ATR% / kapanıştan-kapanışa volatilite (bar içi gürültü oranı)
        vol_ratio      : kısa / uzun EWMA volatilite (volatilite rejimi genişliyor mu?)
        bb_width       : (üst bant - alt bant) / orta bant (sıkışma / genişleme)
        autocorr       : getirilerin 1 gecikmeli otokorelasyonu (+ momentum, - ortalamaya dönüş)
        efficiency     : Kaufman verimlilik oranı (trend kalitesi)
        abs_momentum   : |k-bar log getiri| / (vol * sqrt(k)) (yönsüz trend gücü, z-skoru)
        ret_skew       : getiri çarpıklığı
        volume_z       : log hacim z-skoru
    """
    cfg = cfg or FeatureConfig()
    close = ohlcv["Close"]
    log_ret = np.log(close).diff()

    vol_fast = log_ret.ewm(span=cfg.vol_fast_span, min_periods=cfg.vol_fast_span).std()
    vol_slow = log_ret.ewm(span=cfg.vol_slow_span, min_periods=cfg.vol_slow_span).std()
    atr_pct = average_true_range(ohlcv, cfg.atr_window) / close

    mid = close.rolling(cfg.bb_window).mean()
    sd = close.rolling(cfg.bb_window).std(ddof=0)

    mom = np.log(close / close.shift(cfg.momentum_window))
    log_vol = np.log(ohlcv["Volume"].replace(0.0, np.nan))
    vol_mean = log_vol.rolling(cfg.volume_window).mean()
    vol_std = log_vol.rolling(cfg.volume_window).std()

    features = pd.DataFrame(
        {
            "rsi": rsi(close, cfg.rsi_window),
            "atr_pct": atr_pct,
            "atr_vol_ratio": atr_pct / vol_slow,
            "vol_ratio": vol_fast / vol_slow,
            "bb_width": 2.0 * cfg.bb_num_std * sd / mid,
            "autocorr": log_ret.rolling(cfg.autocorr_window).corr(log_ret.shift(1)),
            "efficiency": efficiency_ratio(close, cfg.efficiency_window),
            "abs_momentum": mom.abs() / (vol_slow * np.sqrt(cfg.momentum_window)),
            "ret_skew": log_ret.rolling(cfg.skew_window).skew(),
            "volume_z": (log_vol - vol_mean) / vol_std.replace(0.0, np.nan),
        },
        index=ohlcv.index,
    )
    return features.replace([np.inf, -np.inf], np.nan)
