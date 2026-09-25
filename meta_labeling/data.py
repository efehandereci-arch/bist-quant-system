"""Veri katmanı: sentetik OHLCV simülasyonu ve CSV yükleyici."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import SimulationConfig

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

TREND, MEAN_REVERT = 0, 1


def simulate_ohlcv(
    cfg: SimulationConfig | None = None,
    return_regimes: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Rejim değiştiren, GARCH volatiliteli sentetik günlük OHLCV serisi üretir.

    Log getiri dinamiği:

    * Trend rejimi      : r_t = s * mu + phi * r_{t-1} + sigma_t * z_t
      (s = rejime girişte rastgele seçilen yön, phi > 0 momentum)
    * Mean-revert rejimi: r_t = -kappa * (x_{t-1} - anchor) + sigma_t * z_t
      (anchor = rejime girişteki log fiyat)

    sigma_t GARCH(1,1) ile güncellenir, z_t birim varyanslı Student-t'dir.
    Gizli rejim serisi yalnızca teşhis amaçlıdır; öznitelik olarak KULLANILMAMALIDIR
    (gerçek piyasada gözlenemez).

    Args:
        cfg: Simülasyon parametreleri.
        return_regimes: True ise (ohlcv, rejim_serisi) döndürülür.
    """
    cfg = cfg or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_bars

    # 1) Gizli Markov rejim zinciri
    regimes = np.empty(n, dtype=np.int8)
    regimes[0] = TREND
    switch = rng.random(n) > cfg.regime_persistence
    for t in range(1, n):
        regimes[t] = 1 - regimes[t - 1] if switch[t] else regimes[t - 1]

    # 2) Student-t şokları (birim varyansa ölçeklenmiş)
    dof = cfg.t_dof
    z = rng.standard_t(dof, size=n) / np.sqrt(dof / (dof - 2.0))

    # 3) GARCH(1,1) + rejime bağlı ortalama denklemi
    var_lr = cfg.long_run_vol**2
    omega = var_lr * (1.0 - cfg.garch_alpha - cfg.garch_beta)
    vol_mult = {TREND: cfg.trend_vol_mult, MEAN_REVERT: cfg.mr_vol_mult}

    log_close = np.empty(n)
    rets = np.empty(n)
    sigmas = np.empty(n)
    x_prev, r_prev, eps_prev, sig2 = np.log(cfg.s0), 0.0, 0.0, var_lr
    trend_sign, anchor = 1.0, x_prev
    for t in range(n):
        if t > 0:
            sig2 = omega + cfg.garch_alpha * eps_prev**2 + cfg.garch_beta * sig2
        if t == 0 or regimes[t] != regimes[t - 1]:
            trend_sign = rng.choice([-1.0, 1.0])
            anchor = x_prev
        sigma = np.sqrt(sig2) * vol_mult[int(regimes[t])]
        if regimes[t] == TREND:
            mu = trend_sign * cfg.trend_drift + cfg.trend_ar * r_prev
        else:
            mu = -cfg.mr_kappa * (x_prev - anchor)
        eps = sigma * z[t]
        r = mu + eps
        x_prev = x_prev + r
        log_close[t], rets[t], sigmas[t] = x_prev, r, sigma
        r_prev, eps_prev = r, eps / vol_mult[int(regimes[t])]

    # 4) Kapanıştan tutarlı OHLC ve hacim üret
    close = np.exp(log_close)
    prev_close = np.concatenate([[cfg.s0], close[:-1]])
    open_ = prev_close * np.exp(rng.normal(0.0, 0.2 * sigmas))
    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    high = body_hi * np.exp(np.abs(rng.normal(0.0, 0.5 * sigmas)))
    low = body_lo * np.exp(-np.abs(rng.normal(0.0, 0.5 * sigmas)))
    volume = 1e6 * np.exp(rng.normal(0.0, 0.3, n)) * (1.0 + np.abs(rets) / sigmas)

    index = pd.bdate_range(cfg.start, periods=n, name="Date")
    ohlcv = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume.round()},
        index=index,
    )
    if return_regimes:
        return ohlcv, pd.Series(regimes, index=index, name="regime")
    return ohlcv


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame'ini doğrular ve normalize eder (sıralı, tekil, tz'siz indeks)."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik OHLCV kolonları: {missing}")
    out = df[OHLCV_COLUMNS].astype(float).dropna()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if (out["Close"] <= 0).any():
        raise ValueError("Kapanış fiyatları pozitif olmalı.")
    return out


def load_ohlcv_csv(path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    """CSV'den OHLCV yükler (ör. BIST hisse verisi). Kolon adları büyük/küçük harf duyarsızdır."""
    raw = pd.read_csv(path)
    rename = {c: c.strip().capitalize() for c in raw.columns}
    raw = raw.rename(columns=rename)
    date_key = date_col.strip().capitalize()
    if date_key not in raw.columns:
        raise ValueError(f"Tarih kolonu bulunamadı: {date_col}")
    raw = raw.set_index(pd.to_datetime(raw.pop(date_key)))
    return validate_ohlcv(raw)
