"""Execution modeli, maliyet modeli ve bar bazlı portföy simülatörü.

Zamanlama varsayımları (açıkça):

* Sinyal zamanı  : t barının KAPANIŞI. Öznitelikler, volatilite ve birincil yön
  bu kapanışa kadar bilinen veriyle hesaplanır.
* ``mode="next_open"`` (varsayılan): emir t+1 barının AÇILIŞINDA gerçekleşir.
  Sinyalin üretildiği kapanış fiyatıyla işlem yapılmaz; bu, "kapanışı görüp
  aynı kapanıştan işlem yapma" iyimserliğini ortadan kaldırır.
* ``mode="close"``: emir sinyal barının kapanışında gerçekleşir (MOC emri
  varsayımı). Karşılaştırma amaçlı; iyimser bir varsayımdır.
* Çıkış: Bariyer t1 kapanışında tespit edilir; çıkış ``next_open`` modunda
  t1+1 açılışında, ``close`` modunda t1 kapanışında yapılır.

Maliyet (tek yön, işlem gören nominal üzerinden oran):

    rate = komisyon + spread/2 + slippage + k * sqrt(emir_tutarı / ADV)

ADV = son ``adv_window`` barın ortalama (kapanış x hacim) değeri, bir bar
gecikmeli (işlem anında bilinen). Sentetik veride hacim de simüle edildiği
için market impact tahmini bir SİMÜLASYON VARSAYIMIDIR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .settings import CostSettings, ExecutionSettings, RiskSettings


@dataclass
class PortfolioResult:
    """Bar bazlı simülasyon çıktısı (``positions`` = bar boyunca tutulan pozisyon)."""

    returns: pd.Series
    positions: pd.Series
    turnover: pd.Series
    costs: pd.Series
    halted_at: pd.Timestamp | None = None

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


def adv_value(ohlcv: pd.DataFrame, window: int) -> pd.Series:
    """Ortalama günlük işlem hacmi (TL), bir bar gecikmeli: t'de yalnızca t-1'e kadarki veri."""
    value = ohlcv["Close"] * ohlcv["Volume"]
    return value.rolling(window, min_periods=window).mean().shift(1).rename("adv")


def fixed_cost_rate(costs: CostSettings) -> float:
    return (costs.commission_bps + costs.spread_bps / 2.0 + costs.slippage_bps) / 1e4


def cost_rate(
    trade_abs: np.ndarray | float,
    adv: np.ndarray | float | None,
    costs: CostSettings,
    capital: float,
) -> np.ndarray:
    """Tek yön maliyet oranı; ``trade_abs`` sermayeye oranla emir büyüklüğüdür."""
    trade_abs = np.asarray(trade_abs, dtype=float)
    rate = np.full(trade_abs.shape, fixed_cost_rate(costs))
    if costs.impact_k > 0 and adv is not None:
        adv = np.broadcast_to(np.asarray(adv, dtype=float), trade_abs.shape)
        ok = np.isfinite(adv) & (adv > 0)
        participation = np.divide(trade_abs * capital, adv, out=np.zeros_like(trade_abs), where=ok)
        rate = rate + costs.impact_k * np.sqrt(participation)
    return rate


def events_to_target(
    index: pd.DatetimeIndex, t0: pd.DatetimeIndex, t1: pd.Series, signal: pd.Series
) -> pd.Series:
    """Olay sinyallerinden karar zamanı bazında hedef pozisyon.

    İşlem i, t0_i kapanışındaki karardan t1_i kapanışındaki çıkış kararına kadar
    ([t0, t1) karar barları) ``signal_i`` hedefler. Aynı anda açık işlemlerin
    sinyalleri ortalanır (AFML Snippet 10.2, avgActiveSignals); böylece brüt
    pozisyon |signal| <= 1 sınırında kalır.
    """
    signal = signal.reindex(t0).fillna(0.0)
    active = signal.to_numpy() != 0
    start = index.get_indexer(t0[active])
    end = index.get_indexer(pd.DatetimeIndex(t1.reindex(t0[active])))
    n = len(index)
    total, count = np.zeros(n + 1), np.zeros(n + 1)
    vals = signal.to_numpy()[active]
    np.add.at(total, start, vals)
    np.add.at(total, end, -vals)
    np.add.at(count, start, 1.0)
    np.add.at(count, end, -1.0)
    total, count = np.cumsum(total[:-1]), np.cumsum(count[:-1])
    target = np.divide(total, count, out=np.zeros(n), where=count > 0.5)
    return pd.Series(target, index=index, name="target")


def simulate_portfolio(
    target: pd.Series,
    ohlcv: pd.DataFrame,
    execution: ExecutionSettings,
    costs: CostSettings,
    risk: RiskSettings,
    adv: pd.Series | None = None,
) -> PortfolioResult:
    """Hedef pozisyon serisini execution + maliyet + risk limitleriyle simüle eder.

    ``target[t]`` t kapanışında verilen karardır. Risk limitleri (hepsi opsiyonel):
      * max_position / max_gross_exposure : hedef pozisyon kırpılır (kaldıraç yok)
      * max_turnover  : bar başına |Δpozisyon| sınırı
      * max_daily_loss: günlük kayıp eşiği aşılırsa ertesi bar pozisyon kapatılır
      * max_drawdown_stop: tepe noktasından düşüş eşiği aşılırsa strateji durdurulur
    """
    index = ohlcv.index
    cap = min(risk.max_position, risk.max_gross_exposure)
    tgt = target.reindex(index).fillna(0.0).clip(-cap, cap).to_numpy()
    close = ohlcv["Close"].to_numpy(dtype=float)
    open_ = ohlcv["Open"].to_numpy(dtype=float)
    adv_arr = adv.reindex(index).to_numpy(dtype=float) if adv is not None else None
    capital = execution.initial_capital
    fixed = fixed_cost_rate(costs)
    next_open = execution.mode == "next_open"

    n = len(index)
    rets, pos_arr, turn, cost_arr = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    pos, equity, peak = 0.0, 1.0, 1.0
    halted_at, flat_next = None, False
    for t in range(1, n):
        desired = tgt[t - 1]
        if halted_at is not None or flat_next:
            desired = 0.0
        if risk.max_turnover is not None:
            desired = pos + float(np.clip(desired - pos, -risk.max_turnover, risk.max_turnover))
        delta = abs(desired - pos)
        rate = fixed
        if delta > 0 and costs.impact_k > 0 and adv_arr is not None:
            a = adv_arr[t]
            if np.isfinite(a) and a > 0:
                rate += costs.impact_k * np.sqrt(delta * capital / a)
        cost = delta * rate
        if next_open:
            r_on = open_[t] / close[t - 1] - 1.0
            r_id = close[t] / open_[t] - 1.0
            r = (1.0 + pos * r_on) * (1.0 + desired * r_id) - 1.0 - cost
        else:
            r = desired * (close[t] / close[t - 1] - 1.0) - cost
        pos = desired
        rets[t], pos_arr[t], turn[t], cost_arr[t] = r, pos, delta, cost
        equity *= 1.0 + r
        peak = max(peak, equity)
        if risk.max_drawdown_stop is not None and halted_at is None:
            if equity / peak - 1.0 <= -abs(risk.max_drawdown_stop):
                halted_at = index[t]
        flat_next = risk.max_daily_loss is not None and r <= -abs(risk.max_daily_loss)

    return PortfolioResult(
        returns=pd.Series(rets, index=index, name="returns"),
        positions=pd.Series(pos_arr, index=index, name="position"),
        turnover=pd.Series(turn, index=index, name="turnover"),
        costs=pd.Series(cost_arr, index=index, name="cost"),
        halted_at=halted_at,
    )


def execution_positions(
    index: pd.DatetimeIndex, t0: pd.DatetimeIndex, t1: pd.Series, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Giriş/çıkış bar pozisyonları (``next_open``: bir sonraki bar)."""
    shift = 1 if mode == "next_open" else 0
    entry = index.get_indexer(t0) + shift
    exit_ = index.get_indexer(pd.DatetimeIndex(t1.reindex(t0))) + shift
    if (entry >= len(index)).any() or (exit_ >= len(index)).any():
        raise ValueError("Execution barı veri sonunun ötesinde")
    return entry, exit_


def trade_ledger(
    ohlcv: pd.DataFrame,
    t1: pd.Series,
    signal: pd.Series,
    execution: ExecutionSettings,
    costs: CostSettings,
    adv: pd.Series | None = None,
) -> pd.DataFrame:
    """İşlem seviyesi defter (maliyet sonrası, işlem nominaline göre getiri).

    Not: Portföy simülasyonunda eşzamanlı işlemler ortalandığı için gerçek emir
    büyüklüğü burada varsayılan |size| değerinden farklı olabilir; defter işlem
    kalitesini (win rate, profit factor, dağılım) ölçmek içindir.
    """
    signal = signal[signal != 0]
    index = ohlcv.index
    if signal.empty:
        return pd.DataFrame(columns=["entry_time", "exit_time", "side", "size", "gross", "cost", "net"])
    entry, exit_ = execution_positions(index, signal.index, t1, execution.mode)
    px = ohlcv["Open" if execution.mode == "next_open" else "Close"].to_numpy(dtype=float)
    size = signal.abs().to_numpy()
    side = np.sign(signal.to_numpy())
    gross = size * side * (px[exit_] / px[entry] - 1.0)
    adv_arr = adv.reindex(index).to_numpy(dtype=float) if adv is not None else None
    cap = execution.initial_capital
    c_in = cost_rate(size, adv_arr[entry] if adv_arr is not None else None, costs, cap)
    c_out = cost_rate(size, adv_arr[exit_] if adv_arr is not None else None, costs, cap)
    cost = size * (c_in + c_out)
    return pd.DataFrame(
        {
            "entry_time": index[entry],
            "exit_time": index[exit_],
            "side": side.astype(int),
            "size": size,
            "gross": gross,
            "cost": cost,
            "net": gross - cost,
        },
        index=signal.index,
    )


def segment_trades(port: PortfolioResult) -> pd.Series:
    """Sürekli pozisyonlu stratejilerde (buy & hold, sürekli EMA kuralı) işlem = aynı işaretli pozisyon bloğu."""
    sign = np.sign(port.positions)
    seg = (sign != sign.shift(1)).cumsum()
    grouped = (1.0 + port.returns[sign != 0]).groupby(seg[sign != 0])
    return (grouped.prod() - 1.0).rename("net").reset_index(drop=True)
