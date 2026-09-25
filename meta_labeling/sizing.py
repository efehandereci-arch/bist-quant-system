"""Olasılıktan pozisyon büyüklüğüne (AFML Bölüm 10).

Meta-modelin P(Y=1) çıktısı iki şekilde kullanılır:

1. Filtre: P(Y=1) > eşik ise birincil sinyale uy, değilse pas geç.
2. Bet sizing: Güven arttıkça pozisyon büyür. Prado, olasılığı bir test
   istatistiğine dönüştürür (H0: p = 1/2):

       z = (p - 1/2) / sqrt(p (1 - p)),     m = 2 Φ(z) - 1  ∈ [0, 1)

   p = 0.5 -> m = 0 (bahis yok), p -> 1 -> m -> 1 (tam pozisyon).
   Nihai sinyal = side x m. İsteğe bağlı ayrıklaştırma (step_size), küçük
   olasılık oynamalarının gereksiz işlem (overtrading) üretmesini engeller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr  # scikit-learn'ün zorunlu bağımlılığı olan scipy'den


def bet_size_from_proba(proba: pd.Series, threshold: float = 0.5, step_size: float = 0.0) -> pd.Series:
    """P(Y=1) -> [0, 1] aralığında pozisyon büyüklüğü; eşik altı 0."""
    p = proba.clip(1e-6, 1 - 1e-6)
    z = (p - 0.5) / np.sqrt(p * (1.0 - p))
    size = pd.Series(2.0 * ndtr(z.to_numpy()) - 1.0, index=proba.index).clip(lower=0.0)
    if step_size > 0:
        size = (size / step_size).round() * step_size
    size[proba <= threshold] = 0.0
    return size.clip(0.0, 1.0).rename("size")


def meta_signals(
    side: pd.Series, proba: pd.Series, threshold: float, step_size: float = 0.0
) -> dict[str, pd.Series]:
    """Karşılaştırılacak üç sinyal seti (hepsi aynı OOS olay kümesi üzerinde).

    * primary : birincil modelin tüm sinyalleri, sabit büyüklük
    * filtered: yalnızca P(Y=1) > eşik olan sinyaller, sabit büyüklük
    * sized   : P(Y=1) > eşik olan sinyaller, büyüklük = m(p)
    """
    taken = (proba > threshold).astype(float)
    return {
        "primary": side.astype(float),
        "filtered": side * taken,
        "sized": side * bet_size_from_proba(proba, threshold, step_size),
    }
