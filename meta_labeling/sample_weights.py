"""Örtüşen etiketler için örnek ağırlıkları (AFML Bölüm 4).

Triple Barrier etiketleri zaman içinde örtüşür: t0=10'da açılıp t1=18'de
kapanan bir işlem ile t0=12'de açılan işlem aynı getirilere bağlıdır. Bu,
gözlemlerin IID olmadığı anlamına gelir. Örtüşen gözlemlere eşit ağırlık
vermek modeli aynı bilginin kopyalarını ezberlemeye (overfitting) iter.

* Eşzamanlılık (concurrency) c_t : t barında aktif olan etiket sayısı.
* Benzersizlik (uniqueness) u_i  : etiketin ömrü boyunca 1/c_t'nin ortalaması.
  Tamamen bağımsız bir etiket için 1, çok kalabalık dönemdeki etiket için ~0.
* Ortalama benzersizlik ayrıca bagging'deki ``max_samples`` için doğal bir
  üst sınırdır: bootstrap örnekleri gereğinden fazla örtüşen gözlem içermez.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _event_positions(index: pd.DatetimeIndex, t1: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    start = index.get_indexer(t1.index)
    end = index.get_indexer(pd.DatetimeIndex(t1.to_numpy()))
    if (start < 0).any() or (end < 0).any():
        raise KeyError("Olay başlangıç/bitiş zamanları fiyat indeksinde olmalı")
    return start, end


def num_concurrent_events(index: pd.DatetimeIndex, t1: pd.Series) -> pd.Series:
    """Her bar için aktif (ömrü [t0, t1] aralığında o barı kapsayan) etiket sayısı (Snippet 4.1)."""
    start, end = _event_positions(index, t1)
    diff = np.zeros(len(index) + 1)
    np.add.at(diff, start, 1.0)
    np.add.at(diff, end + 1, -1.0)
    return pd.Series(np.cumsum(diff[:-1]), index=index, name="num_co_events")


def average_uniqueness(
    index: pd.DatetimeIndex, t1: pd.Series, num_co_events: pd.Series | None = None
) -> pd.Series:
    """Her etiketin ortalama benzersizliği: mean_{t ∈ [t0, t1]} 1 / c_t (Snippet 4.2)."""
    if num_co_events is None:
        num_co_events = num_concurrent_events(index, t1)
    start, end = _event_positions(index, t1)
    c = num_co_events.to_numpy(dtype=float)
    inv = np.divide(1.0, c, out=np.zeros_like(c), where=c > 0)
    cum = np.concatenate([[0.0], np.cumsum(inv)])
    avg_u = (cum[end + 1] - cum[start]) / (end - start + 1)
    return pd.Series(avg_u, index=t1.index, name="uniqueness")


def return_attribution_weights(
    close: pd.Series, t1: pd.Series, num_co_events: pd.Series | None = None
) -> pd.Series:
    """Getiri atfı ile ağırlıklandırma (Snippet 4.10).

    w_i ∝ | Σ_{t ∈ (t0, t1]} r_t / c_t |  -- büyük ve benzersiz mutlak getiriye
    sahip etiketler daha fazla ağırlık alır. Ağırlıklar toplamı gözlem sayısına
    eşit olacak şekilde normalize edilir.
    """
    index = close.index
    if num_co_events is None:
        num_co_events = num_concurrent_events(index, t1)
    start, end = _event_positions(index, t1)
    ret = np.log(close).diff().fillna(0.0).to_numpy()
    c = num_co_events.to_numpy(dtype=float)
    attributed = np.divide(ret, c, out=np.zeros_like(ret), where=c > 0)
    cum = np.concatenate([[0.0], np.cumsum(attributed)])
    w = np.abs(cum[end + 1] - cum[start + 1])
    w = w * len(w) / w.sum() if w.sum() > 0 else np.ones_like(w)
    return pd.Series(w, index=t1.index, name="return_weight")
