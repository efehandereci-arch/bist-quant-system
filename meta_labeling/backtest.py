"""Olay tabanlı basit backtest ve performans metrikleri.

İki seviyede ölçüm yapılır:

* İşlem seviyesi: Her olay için t0 kapanışında giriş, t1 (ilk bariyer) kapanışında
  çıkış. Win rate, ortalama getiri, işlem Sharpe'ı.
* Portföy seviyesi: Aynı anda açık olan işlemlerin sinyalleri ortalanır
  (AFML Snippet 10.2 ``avgActiveSignals``) ve bar bazında pozisyon oluşturulur.
  Sharpe oranı bu günlük getiri serisi üzerinden yıllıklandırılır. Örtüşen
  işlemlerde işlem Sharpe'ı şişebileceği için asıl referans budur.

Ek olarak Probabilistic Sharpe Ratio (PSR; Bailey & López de Prado, 2012)
raporlanır: Gözlenen Sharpe'ın, getirilerin çarpıklık ve basıklığı hesaba
katıldığında 0'dan büyük olma olasılığı.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.special import ndtr


@dataclass(frozen=True)
class StrategyStats:
    n_trades: int
    win_rate: float
    avg_trade_ret: float
    trade_sharpe: float
    ann_return: float
    ann_vol: float
    sharpe: float
    psr: float
    max_drawdown: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def trade_returns(
    close: pd.Series, t1: pd.Series, signal: pd.Series, cost_per_side: float = 0.0
) -> pd.Series:
    """Sinyali sıfırdan farklı olayların maliyet sonrası işlem getirileri."""
    signal = signal[signal != 0]
    p0 = close.reindex(signal.index).to_numpy()
    p1 = close.reindex(pd.DatetimeIndex(t1.reindex(signal.index))).to_numpy()
    gross = signal.to_numpy() * (p1 / p0 - 1.0)
    net = gross - 2.0 * cost_per_side * signal.abs().to_numpy()
    return pd.Series(net, index=signal.index, name="trade_ret")


def average_active_positions(index: pd.DatetimeIndex, t1: pd.Series, signal: pd.Series) -> pd.Series:
    """Bar bazında pozisyon = o bar boyunca açık olan işlemlerin sinyal ortalaması.

    t0'da kapanışta açılan işlem, (t0, t1] aralığındaki bar getirilerine maruz kalır.
    """
    signal = signal[signal != 0]
    start = index.get_indexer(signal.index) + 1
    end = index.get_indexer(pd.DatetimeIndex(t1.reindex(signal.index)))
    n = len(index)
    total, count = np.zeros(n + 1), np.zeros(n + 1)
    np.add.at(total, start, signal.to_numpy())
    np.add.at(total, end + 1, -signal.to_numpy())
    np.add.at(count, start, 1.0)
    np.add.at(count, end + 1, -1.0)
    total, count = np.cumsum(total[:-1]), np.cumsum(count[:-1])
    pos = np.divide(total, count, out=np.zeros(n), where=count > 0.5)
    return pd.Series(pos, index=index, name="position")


def portfolio_returns(
    close: pd.Series, t1: pd.Series, signal: pd.Series, cost_per_side: float = 0.0
) -> pd.Series:
    """Bar bazında strateji getirisi: pozisyon x bar getirisi - maliyet x |Δpozisyon|."""
    pos = average_active_positions(close.index, t1, signal)
    bar_ret = close.pct_change().fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    return (pos * bar_ret - cost_per_side * turnover).rename("strategy_ret")


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR = Φ( (SR - SR*) sqrt(n-1) / sqrt(1 - γ3 SR + (γ4 - 1)/4 SR²) ), periyot bazında SR."""
    r = returns.dropna()
    n, sd = len(r), r.std()
    if n < 3 or sd == 0:
        return float("nan")
    sr = r.mean() / sd
    skew, kurt = r.skew(), r.kurt() + 3.0  # pandas kurt() fazlalık basıklık döndürür
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    return float(ndtr((sr - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom)))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns).cumprod()
    return float((equity / equity.cummax() - 1.0).min())


def evaluate_strategy(
    close: pd.Series,
    t1: pd.Series,
    signal: pd.Series,
    cost_per_side: float = 0.0,
    periods_per_year: int = 252,
    window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> StrategyStats:
    """Tek bir sinyal seti için işlem ve portföy seviyesi metrikler.

    ``window`` verilirse portföy getirileri bu aralıkla sınırlanır; farklı
    stratejiler aynı takvim üzerinde (boşta geçen günler dahil) karşılaştırılır.
    """
    trades = trade_returns(close, t1, signal, cost_per_side)
    port = portfolio_returns(close, t1, signal, cost_per_side)
    if window is not None:
        port = port.loc[window[0] : window[1]]
    years = max(len(port) / periods_per_year, 1e-9)

    if len(trades) > 1 and trades.std() > 0:
        trade_sharpe = trades.mean() / trades.std() * np.sqrt(len(trades) / years)
    else:
        trade_sharpe = float("nan")
    ann_ret = port.mean() * periods_per_year
    ann_vol = port.std() * np.sqrt(periods_per_year)
    return StrategyStats(
        n_trades=len(trades),
        win_rate=float((trades > 0).mean()) if len(trades) else float("nan"),
        avg_trade_ret=float(trades.mean()) if len(trades) else float("nan"),
        trade_sharpe=float(trade_sharpe),
        ann_return=float(ann_ret),
        ann_vol=float(ann_vol),
        sharpe=float(ann_ret / ann_vol) if ann_vol > 0 else float("nan"),
        psr=probabilistic_sharpe_ratio(port),
        max_drawdown=max_drawdown(port),
    )


def compare_strategies(
    close: pd.Series,
    t1: pd.Series,
    signals: dict[str, pd.Series],
    cost_per_side: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Birden çok sinyal setini aynı OOS penceresinde karşılaştırır."""
    window = (t1.index.min(), t1.max())
    rows = {
        name: evaluate_strategy(close, t1, sig, cost_per_side, periods_per_year, window).as_dict()
        for name, sig in signals.items()
    }
    return pd.DataFrame(rows).T
