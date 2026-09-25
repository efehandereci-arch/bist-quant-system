"""Dinamik volatilite tahmini (AFML Snippet 3.1).

Prado, bariyerleri sabit yüzdelerle (ör. %2 kâr al / %2 zarar kes) belirlemenin
hatalı olduğunu vurgular: Volatilitenin yüksek olduğu dönemlerde sabit bariyerler
gürültüyle tetiklenir, düşük olduğu dönemlerde ise hiç tetiklenmez. Bu yüzden
bariyer genişliği ("trgt") olay anındaki *tahmini* volatiliteye göre ölçeklenir.
Tahmin yalnızca t anına kadarki getirileri kullanır (look-ahead yok).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def get_volatility(close: pd.Series, span: int = 50, method: str = "ewma") -> pd.Series:
    """Bar başına log getiri volatilitesi.

    Args:
        close: Kapanış fiyatları.
        span: EWMA span'i (``method="ewma"``) veya pencere uzunluğu (``"rolling"``).
        method: ``"ewma"`` (Prado'nun tercihi, yakın geçmişe daha fazla ağırlık)
            veya ``"rolling"`` (eşit ağırlıklı standart sapma).
    """
    ret = log_returns(close)
    min_periods = max(10, span // 2)
    if method == "ewma":
        vol = ret.ewm(span=span, min_periods=min_periods).std()
    elif method == "rolling":
        vol = ret.rolling(span, min_periods=min_periods).std()
    else:
        raise ValueError(f"Bilinmeyen volatilite yöntemi: {method}")
    return vol.rename("vol")
