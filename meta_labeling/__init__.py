"""Meta-Labeling + Triple Barrier Method ile aşırı öğrenmeye dirençli ML işlem pipeline'ı.

Marcos López de Prado, *Advances in Financial Machine Learning* (2018)
metodolojisini izler:

* Bölüm 2  : CUSUM filtresi ile olay örnekleme           -> ``sampling``
* Bölüm 3  : Dinamik volatilite, Triple Barrier, meta-etiket -> ``volatility``, ``labeling``
* Bölüm 4  : Örtüşen etiketler için benzersizlik ağırlıkları -> ``sample_weights``
* Bölüm 6-7: Bagging ayarları, Purged & Embargoed CV      -> ``model``, ``cv``
* Bölüm 10 : Olasılıktan bet sizing                      -> ``sizing``
* Bölüm 14 : Sharpe / Probabilistic Sharpe Ratio         -> ``backtest``
"""

from .config import (
    BarrierConfig,
    CVConfig,
    FeatureConfig,
    MetaModelConfig,
    PipelineConfig,
    PrimaryConfig,
    SimulationConfig,
)
from .cv import PurgedKFold, cross_validate_purged, walk_forward_predict
from .data import load_ohlcv_csv, simulate_ohlcv
from .labeling import apply_triple_barrier, get_meta_labels, get_vertical_barriers
from .pipeline import MetaLabelingPipeline, PipelineResult

__all__ = [
    "BarrierConfig",
    "CVConfig",
    "FeatureConfig",
    "MetaLabelingPipeline",
    "MetaModelConfig",
    "PipelineConfig",
    "PipelineResult",
    "PrimaryConfig",
    "PurgedKFold",
    "SimulationConfig",
    "apply_triple_barrier",
    "cross_validate_purged",
    "get_meta_labels",
    "get_vertical_barriers",
    "load_ohlcv_csv",
    "simulate_ohlcv",
    "walk_forward_predict",
]
