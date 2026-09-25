"""Performans metrikleri, drawdown analizi, PSR/DSR ve bootstrap.

İstatistiksel notlar
--------------------
* PSR (Bailey & López de Prado, 2012): Gözlenen Sharpe'ın, getirilerin
  çarpıklık ve basıklığı hesaba katılarak, bir eşik Sharpe'tan (varsayılan 0)
  büyük olma olasılığı.
* DSR (Bailey & López de Prado, 2014): PSR'nin, N deneme arasından "en iyisini
  seçme" yanlılığına göre düzeltilmiş hali. Eşik Sharpe, N denemenin Sharpe
  varyansından türetilen beklenen maksimum Sharpe'tır. N, bu çalıştırmada
  OOS üzerinde değerlendirilen tüm strateji varyantlarının sayısıdır.
* Bootstrap: Finansal getiriler IID değildir (volatilite kümelenmesi,
  otokorelasyon, örtüşen işlemler). IID bootstrap bu bağımlılığı yok ederek
  güven aralıklarını olduğundan DAR gösterebilir. Bu yüzden birincil sonuç
  olarak durağan blok bootstrap (Politis & Romano, 1994) raporlanır; IID
  sonuçlar yalnızca karşılaştırma içindir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtri

from ..backtest import probabilistic_sharpe_ratio
from .execution import PortfolioResult

EULER_GAMMA = 0.5772156649015329


def sharpe(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def cagr(returns: pd.Series | np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    years = len(r) / periods_per_year
    return growth ** (1.0 / years) - 1.0 if growth > 0 else -1.0


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min()) if len(equity) else float("nan")


def profit_factor(trade_rets: pd.Series) -> float:
    gains = trade_rets[trade_rets > 0].sum()
    losses = -trade_rets[trade_rets < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf") if gains > 0 else float("nan")


def performance_metrics(
    port: PortfolioResult, trade_rets: pd.Series, periods_per_year: int = 252
) -> dict[str, float]:
    """Tek bir stratejinin standart metrik seti (tüm stratejiler için aynı tanımlar)."""
    r = port.returns
    years = max(len(r) / periods_per_year, 1e-9)
    total = float((1.0 + r).prod() - 1.0)
    g = cagr(r, periods_per_year)
    ann_vol = float(r.std() * np.sqrt(periods_per_year))
    downside = float(np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2)) * np.sqrt(periods_per_year))
    mdd = max_drawdown(r)
    return {
        "Trades": int(len(trade_rets)),
        "CAGR": g,
        "Total Return": total,
        "Ann. Volatility": ann_vol,
        "Sharpe": sharpe(r, periods_per_year),
        "Sortino": float(r.mean() * periods_per_year / downside) if downside > 0 else float("nan"),
        "Max Drawdown": mdd,
        "Calmar": float(g / abs(mdd)) if mdd < 0 else float("nan"),
        "Win Rate": float((trade_rets > 0).mean()) if len(trade_rets) else float("nan"),
        "Profit Factor": profit_factor(trade_rets),
        "Turnover": float(port.turnover.sum() / years),
        "Exposure": float((port.positions.abs() > 1e-12).mean()),
        "PSR": probabilistic_sharpe_ratio(r),
        "DSR": float("nan"),
    }


# ------------------------------------------------------------------ PSR / DSR
def psr(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """Periyot bazında (yıllıklandırılmamış) eşik Sharpe'a göre PSR."""
    return probabilistic_sharpe_ratio(returns, sr_benchmark)


def expected_max_sharpe(trial_srs: np.ndarray, n_trials: int) -> float:
    """N bağımsız denemenin beklenen maksimum Sharpe'ı (periyot bazında)."""
    if n_trials < 2 or len(trial_srs) < 2:
        return 0.0
    sd = float(np.std(trial_srs, ddof=1))
    return sd * (
        (1.0 - EULER_GAMMA) * ndtri(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * ndtri(1.0 - 1.0 / (n_trials * np.e))
    )


def deflated_sharpe(returns: pd.Series, trial_srs: np.ndarray, n_trials: int | None = None) -> float:
    """DSR = PSR(SR* = E[max SR]); ``trial_srs`` periyot bazında Sharpe'lardır."""
    n = n_trials if n_trials is not None else len(trial_srs)
    return psr(returns, expected_max_sharpe(np.asarray(trial_srs, dtype=float), n))


def per_period_sharpe(returns: pd.Series) -> float:
    sd = returns.std()
    return float(returns.mean() / sd) if sd > 0 else 0.0


# -------------------------------------------------------------- drawdown
def drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    """Her drawdown dönemi: başlangıç (tepe), dip, toparlanma, derinlik, süre, toparlanma süresi (bar)."""
    equity = (1.0 + returns).cumprod()
    dd = equity / equity.cummax() - 1.0
    under = dd < 0
    rows = []
    idx = returns.index
    i, n = 0, len(dd)
    while i < n:
        if not under.iloc[i]:
            i += 1
            continue
        start = i
        while i < n and under.iloc[i]:
            i += 1
        end = i  # toparlanma barı (n ise henüz toparlanmadı)
        trough = start + int(np.argmin(dd.iloc[start:end].to_numpy()))
        rows.append(
            {
                "peak": idx[start - 1] if start > 0 else idx[start],
                "trough": idx[trough],
                "recovery": idx[end] if end < n else pd.NaT,
                "depth": float(dd.iloc[trough]),
                "duration_bars": end - start,
                "recovery_bars": (end - trough) if end < n else np.nan,
                "recovered": end < n,
            }
        )
    return pd.DataFrame(rows)


def drawdown_summary(returns: pd.Series, top: int = 5) -> tuple[dict[str, float], pd.DataFrame]:
    ep = drawdown_episodes(returns)
    if ep.empty:
        return {"Max Drawdown": 0.0}, ep
    stats = {
        "Max Drawdown": float(ep["depth"].min()),
        "Average Drawdown": float(ep["depth"].mean()),
        "Median Drawdown": float(ep["depth"].median()),
        "Episodes": int(len(ep)),
        "Avg Duration (bars)": float(ep["duration_bars"].mean()),
        "Max Duration (bars)": float(ep["duration_bars"].max()),
        "Avg Recovery (bars)": float(ep["recovery_bars"].mean()),
        "Unrecovered at end": bool(not ep["recovered"].iloc[-1]),
    }
    return stats, ep.nsmallest(top, "depth").reset_index(drop=True)


def trade_distribution(trade_rets: pd.Series) -> dict[str, float]:
    t = trade_rets.dropna()
    if t.empty:
        return {}
    return {
        "Mean": float(t.mean()),
        "Median": float(t.median()),
        "Std": float(t.std()),
        "Skew": float(t.skew()),
        "Kurtosis": float(t.kurt()),
        "Min": float(t.min()),
        "Max": float(t.max()),
        "Profit Factor": profit_factor(t),
        "Avg Win": float(t[t > 0].mean()) if (t > 0).any() else float("nan"),
        "Avg Loss": float(t[t < 0].mean()) if (t < 0).any() else float("nan"),
    }


# -------------------------------------------------------------- bootstrap
def stationary_bootstrap_indices(n: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano durağan bootstrap: geometrik uzunluklu, dairesel bloklar."""
    p = 1.0 / max(mean_block, 1)
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    new_block = rng.random(n) < p
    starts = rng.integers(0, n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % n
    return idx


def bootstrap_returns(
    returns: pd.Series,
    n_boot: int,
    rng: np.random.Generator,
    block: int | None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Sharpe ve CAGR için bootstrap dağılımı. ``block=None`` -> IID bootstrap."""
    r = returns.to_numpy(dtype=float)
    n = len(r)
    srs, cagrs = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(n, block, rng) if block else rng.integers(0, n, n)
        sample = r[idx]
        srs[b] = sharpe(sample, periods_per_year)
        cagrs[b] = cagr(sample, periods_per_year)
    return {
        "sharpe_ci_low": float(np.nanpercentile(srs, 2.5)),
        "sharpe_median": float(np.nanmedian(srs)),
        "sharpe_ci_high": float(np.nanpercentile(srs, 97.5)),
        "p_sharpe_gt_0": float(np.nanmean(srs > 0)),
        "p_cagr_lt_0": float(np.nanmean(cagrs < 0)),
    }


def bootstrap_trades(
    trade_rets: pd.Series, n_boot: int, rng: np.random.Generator, trades_per_year: float
) -> dict[str, float]:
    """İşlem getirilerinin IID bootstrap'ı (işlem Sharpe'ı, yıllıklandırılmış)."""
    t = trade_rets.to_numpy(dtype=float)
    if len(t) < 3:
        return {}
    srs = np.empty(n_boot)
    for b in range(n_boot):
        s = t[rng.integers(0, len(t), len(t))]
        sd = s.std(ddof=1)
        srs[b] = s.mean() / sd * np.sqrt(trades_per_year) if sd > 0 else np.nan
    return {
        "sharpe_ci_low": float(np.nanpercentile(srs, 2.5)),
        "sharpe_median": float(np.nanmedian(srs)),
        "sharpe_ci_high": float(np.nanpercentile(srs, 97.5)),
        "p_sharpe_gt_0": float(np.nanmean(srs > 0)),
        "p_cagr_lt_0": float("nan"),
    }
