"""Olay (event) örnekleme: hangi barlarda bir işlem kararı değerlendirilecek?

Prado (AFML 2.5) her barı bir gözlem olarak kullanmak yerine *bilgi içeren*
anların örneklenmesini önerir. Her bar etiketlenirse gözlemler aşırı derecede
örtüşür ve IID varsayımı daha da bozulur. Simetrik CUSUM filtresi, fiyatın
son sıfırlamadan bu yana eşik kadar yukarı ya da aşağı sürüklendiği anları
yakalar; yani yalnızca "anlamlı" hareketlerde bir olay üretir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cusum_filter(close: pd.Series, threshold: float | pd.Series) -> pd.DatetimeIndex:
    """Simetrik CUSUM filtresi (AFML Snippet 2.4).

    S+_t = max(0, S+_{t-1} + r_t),  S-_t = min(0, S-_{t-1} + r_t)
    S+_t > h_t veya S-_t < -h_t olduğunda bir olay kaydedilir ve ilgili toplam sıfırlanır.

    Args:
        close: Kapanış fiyatları.
        threshold: Sabit eşik ya da bar bazında dinamik eşik serisi (ör. k x volatilite).
            Dinamik eşik, t anına kadar bilinen bilgiyle hesaplanmalıdır.
    """
    ret = np.log(close).diff().to_numpy()
    if isinstance(threshold, pd.Series):
        thr = threshold.reindex(close.index).to_numpy(dtype=float)
    else:
        thr = np.full(len(close), float(threshold))

    events: list[int] = []
    s_pos = s_neg = 0.0
    for i in range(1, len(ret)):
        r, h = ret[i], thr[i]
        if np.isnan(r) or np.isnan(h):
            continue
        s_pos = max(0.0, s_pos + r)
        s_neg = min(0.0, s_neg + r)
        if s_neg < -h:
            s_neg = 0.0
            events.append(i)
        elif s_pos > h:
            s_pos = 0.0
            events.append(i)
    return close.index[events]


def sample_events(
    side: pd.Series,
    close: pd.Series,
    mode: str = "cusum",
    cusum_threshold: float | pd.Series | None = None,
) -> pd.DatetimeIndex:
    """Birincil modelin yön serisinden işlem olaylarını (t0) seçer.

    * ``"cusum"``  : CUSUM olayları ∩ {side != 0}. Yön olay anındaki birincil sinyaldir.
    * ``"signal"`` : side'ın sıfırdan farklı yeni bir değere geçtiği barlar
      (yeni giriş veya yön değişimi).
    """
    side = side.reindex(close.index).fillna(0)
    if mode == "cusum":
        if cusum_threshold is None:
            raise ValueError("cusum modu için cusum_threshold gerekli")
        t_events = cusum_filter(close, cusum_threshold)
        return t_events[side.loc[t_events].to_numpy() != 0]
    if mode == "signal":
        changed = (side != side.shift(1)) & (side != 0)
        return side.index[changed.to_numpy()]
    raise ValueError(f"Bilinmeyen olay modu: {mode}")
