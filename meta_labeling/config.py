"""Pipeline konfigürasyonu.

Tüm hiperparametreler tek bir yerde, değiştirilemez (frozen) dataclass'lar
olarak tutulur.

Prado'nun "backtest overfitting" uyarısı (AFML Bölüm 11-12): Aynı veri üzerinde
denenen her ek konfigürasyon (farklı eşik, farklı bariyer katsayısı, farklı
model), bulunan "en iyi" stratejinin Sharpe oranının şans eseri yüksek çıkma
olasılığını artırır. Bu nedenle parametreler burada *ex-ante* sabitlenir ve
özellikle karar eşiği (``MetaModelConfig.threshold``) test verisine bakılarak
optimize EDİLMEZ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SimulationConfig:
    """Sentetik OHLCV üreticisinin parametreleri.

    Üretilen seri iki gizli rejim arasında geçiş yapan (Markov zinciri) bir
    süreçtir: *trend* rejiminde getiriler kalıcı bir drift ve pozitif
    otokorelasyon taşır, *ortalamaya dönüş* (mean-reversion) rejiminde fiyat bir
    çapa seviyesine geri çekilir. Volatilite GARCH(1,1), şoklar Student-t
    dağılımlıdır (kalın kuyruk). Böylece trend takipçisi bir birincil model bazı
    rejimlerde kazanır, bazılarında kaybeder; meta-modelin öğrenebileceği
    gerçekçi bir yapı oluşur.
    """

    n_bars: int = 4000
    start: str = "2010-01-04"
    s0: float = 100.0
    seed: int = 42
    regime_persistence: float = 0.99   # P(rejim_{t+1} = rejim_t) -> ort. süre ~100 bar
    trend_drift: float = 0.0015        # trend rejiminde günlük log drift (yönü rastgele)
    trend_ar: float = 0.10             # trend rejiminde getiri AR(1) katsayısı
    mr_kappa: float = 0.08             # ortalamaya dönüş hızı (yarı ömür ~ ln2/kappa bar)
    long_run_vol: float = 0.015        # GARCH uzun dönem günlük volatilitesi
    garch_alpha: float = 0.08
    garch_beta: float = 0.90
    trend_vol_mult: float = 0.90       # rejime bağlı volatilite çarpanları
    mr_vol_mult: float = 1.10
    t_dof: float = 5.0                 # Student-t serbestlik derecesi


@dataclass(frozen=True)
class PrimaryConfig:
    """Birincil (yön) modeli ve olay örnekleme ayarları.

    ``event_mode``:
      * ``"cusum"``  : Olaylar simetrik CUSUM filtresiyle örneklenir (AFML 2.5.2.1),
        yön birincil modelden okunur. Yüksek recall sağlar: kayda değer her fiyat
        hareketinde bir pozisyon önerisi üretilir.
      * ``"signal"`` : Olay yalnızca birincil sinyalin yeni başladığı / yön
        değiştirdiği barlarda oluşur (klasik "giriş sinyali").
    """

    kind: Literal["ema", "bollinger"] = "ema"
    ema_fast: int = 10
    ema_slow: int = 40
    ema_neutral_band: float = 0.0      # |fast-slow|/slow bu değerin altındaysa 0 (nötr)
    bb_window: int = 20
    bb_num_std: float = 1.5            # düşük katsayı -> daha çok sinyal -> daha yüksek recall
    event_mode: Literal["cusum", "signal"] = "cusum"
    cusum_vol_mult: float = 1.0        # CUSUM eşiği = mult x dinamik volatilite


@dataclass(frozen=True)
class BarrierConfig:
    """Triple Barrier Method parametreleri (AFML Bölüm 3)."""

    vol_method: Literal["ewma", "rolling"] = "ewma"
    vol_span: int = 50                 # EWMA span'i ya da rolling pencere uzunluğu
    pt_mult: float = 2.0               # üst bariyer = pt_mult x volatilite (0 -> devre dışı)
    sl_mult: float = 2.0               # alt bariyer = sl_mult x volatilite (0 -> devre dışı)
    max_holding_bars: int = 10         # dikey bariyer (bar cinsinden)
    min_target: float = 0.002          # bu volatilitenin altındaki olaylar atılır


@dataclass(frozen=True)
class FeatureConfig:
    """Meta-model için rejim özniteliklerinin pencere uzunlukları."""

    rsi_window: int = 14
    atr_window: int = 14
    bb_window: int = 20
    bb_num_std: float = 2.0
    vol_fast_span: int = 10
    vol_slow_span: int = 50
    autocorr_window: int = 50
    efficiency_window: int = 20
    momentum_window: int = 20
    skew_window: int = 50
    volume_window: int = 20


@dataclass(frozen=True)
class CVConfig:
    """Purged & embargoed çapraz doğrulama ayarları (AFML Bölüm 7)."""

    n_splits: int = 6
    embargo_pct: float = 0.01          # toplam zaman aralığının %1'i kadar embargo
    min_train_events: int = 100        # walk-forward'da bundan az eğitim olayı varsa fold atlanır


@dataclass(frozen=True)
class MetaModelConfig:
    """İkincil (meta) model, bet sizing ve maliyet ayarları."""

    kind: Literal["lgbm", "rf"] = "lgbm"
    threshold: float = 0.55            # P(Y=1) > threshold ise işlem alınır
    step_size: float = 0.0             # bet size ayrıklaştırma adımı (0 -> sürekli)
    use_side_feature: bool = True      # birincil yönü de öznitelik olarak ver
    weighting: Literal["uniqueness", "return", "none"] = "uniqueness"
    cost_per_side: float = 0.0005      # tek yön işlem maliyeti (5 bps)
    # LightGBM (sığ ağaçlar + güçlü düzenlileştirme = düşük varyans)
    n_estimators: int = 300
    learning_rate: float = 0.02
    num_leaves: int = 8
    max_depth: int = 3
    min_child_samples: int = 40
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    # RandomForest (Prado'nun önerdiği bagging ayarları, AFML 6.4)
    rf_n_estimators: int = 500
    rf_min_weight_fraction_leaf: float = 0.05
    seed: int = 42
    n_jobs: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    primary: PrimaryConfig = field(default_factory=PrimaryConfig)
    barrier: BarrierConfig = field(default_factory=BarrierConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    model: MetaModelConfig = field(default_factory=MetaModelConfig)
    periods_per_year: int = 252
