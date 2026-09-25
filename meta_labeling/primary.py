"""Birincil (primary) model: yalnızca işlemin YÖNÜNÜ belirler.

Meta-labeling (AFML 3.6) iş bölümüne dayanır:

* Birincil model -> "Hangi yönde?" (side ∈ {-1, 0, +1}). Yüksek *recall*
  hedeflenir: fırsatların büyük kısmını yakalamalı, yanlış pozitifler kabul
  edilebilir.
* İkincil (meta) model -> "Bu sinyale uymalı mıyım, ne kadar büyüklükte?"
  Yanlış pozitifleri eleyerek *precision*'ı yükseltir.

Bu ayrım, ML modelinin yön tahmini gibi zor bir problemi çözmesi yerine daha
kolay bir ikili sınıflandırma problemi (sinyal doğru mu?) çözmesini sağlar ve
aşırı öğrenme riskini azaltır. Birincil model bir kural, bir ekonometrik model
ya da bir insan (discretionary trader) olabilir.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PrimaryConfig


class PrimaryModel(ABC):
    """Birincil model arayüzü. ``side`` her bar kapanışında bilinen yönü döndürür."""

    name: str = "primary"

    @abstractmethod
    def side(self, ohlcv: pd.DataFrame) -> pd.Series:
        """Her bar için {-1, 0, +1} yön serisi (yalnızca geçmiş veriye dayanır)."""


@dataclass
class EMACrossover(PrimaryModel):
    """Trend takipçisi EMA kesişimi: hızlı EMA > yavaş EMA ise Long, tersi Short.

    ``neutral_band`` > 0 ise iki EMA arasındaki göreli fark bu bandın altındayken
    nötr (0) kalınır; bu, kesişim anlarındaki whipsaw'ları azaltır.
    """

    fast: int = 10
    slow: int = 40
    neutral_band: float = 0.0

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError("fast < slow olmalı")
        self.name = f"EMA({self.fast}/{self.slow})"

    def side(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["Close"]
        fast = close.ewm(span=self.fast, adjust=False, min_periods=self.fast).mean()
        slow = close.ewm(span=self.slow, adjust=False, min_periods=self.slow).mean()
        spread = (fast - slow) / slow
        side = np.sign(spread).where(spread.abs() > self.neutral_band, 0.0)
        return side.fillna(0.0).astype(int).rename("side")


@dataclass
class BollingerBreakout(PrimaryModel):
    """Bollinger bant kırılımı (trend takipçisi).

    Kapanış üst bandın üzerindeyse +1, alt bandın altındaysa -1, bant içindeyse 0.
    Dar bant (düşük ``num_std``) daha fazla sinyal ve daha yüksek recall demektir.
    """

    window: int = 20
    num_std: float = 1.5

    def __post_init__(self) -> None:
        self.name = f"Bollinger({self.window},{self.num_std})"

    def side(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["Close"]
        mid = close.rolling(self.window).mean()
        sd = close.rolling(self.window).std(ddof=0)
        upper, lower = mid + self.num_std * sd, mid - self.num_std * sd
        side = pd.Series(0, index=close.index, dtype=int)
        side[close > upper] = 1
        side[close < lower] = -1
        return side.rename("side")


def build_primary_model(cfg: PrimaryConfig) -> PrimaryModel:
    if cfg.kind == "ema":
        return EMACrossover(cfg.ema_fast, cfg.ema_slow, cfg.ema_neutral_band)
    if cfg.kind == "bollinger":
        return BollingerBreakout(cfg.bb_window, cfg.bb_num_std)
    raise ValueError(f"Bilinmeyen birincil model: {cfg.kind}")
