"""YAML tabanlı araştırma konfigürasyonu.

Tüm kritik parametreler ``config.yaml`` dosyasından okunur ve değiştirilemez
dataclass'lara dönüştürülür. Bilinmeyen anahtarlar hata üretir (yazım hatası
yüzünden bir parametrenin sessizce varsayılana düşmesi engellenir).

Kural: Bu dosyadaki değerler ex-ante belirlenir. OOS sonuçlarına bakılarak
değiştirilen her parametre yeni bir "deneme" (trial) demektir ve Deflated
Sharpe Ratio hesabında hesaba katılmalıdır.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from ..config import (
    BarrierConfig,
    CVConfig,
    FeatureConfig,
    MetaModelConfig,
    PipelineConfig,
    PrimaryConfig,
    SimulationConfig,
)


class ConfigError(ValueError):
    """Geçersiz ya da eksik konfigürasyon."""


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "meta_labeling_research"
    seed: int = 42
    output_dir: str = "research_output"
    log_level: str = "INFO"
    save_figures: bool = True
    strict_leakage: bool = True          # leakage tespit edilirse LeakageError fırlat


@dataclass(frozen=True)
class DataSettings:
    source: Literal["simulated", "csv", "yfinance"] = "simulated"
    tickers: tuple[str, ...] = ("SASA",)
    csv_dir: str = "data"                # {csv_dir}/{TICKER}.csv
    yfinance_suffix: str = ".IS"
    yfinance_period: str = "15y"
    start: str | None = None
    end: str | None = None
    constituents_file: str | None = None  # tarihsel endeks üyeliği (survivorship kontrolü)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


@dataclass(frozen=True)
class PrimarySettings:
    """Birincil (yön) modeli. Olay örnekleme alanları ``cusum`` bölümündedir."""

    kind: Literal["ema", "bollinger"] = "ema"
    ema_fast: int = 10
    ema_slow: int = 40
    ema_neutral_band: float = 0.0
    bb_window: int = 20
    bb_num_std: float = 1.5


@dataclass(frozen=True)
class CusumSettings:
    event_mode: Literal["cusum", "signal"] = "cusum"
    vol_mult: float = 1.0


@dataclass(frozen=True)
class MetaModelSettings:
    kind: Literal["lgbm", "rf"] = "lgbm"
    threshold: float = 0.55
    use_side_feature: bool = True
    weighting: Literal["uniqueness", "return", "none"] = "uniqueness"
    n_estimators: int = 300
    learning_rate: float = 0.02
    num_leaves: int = 8
    max_depth: int = 3
    min_child_samples: int = 40
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    rf_n_estimators: int = 500
    rf_min_weight_fraction_leaf: float = 0.05
    n_jobs: int = 1


@dataclass(frozen=True)
class CPCVSettings:
    n_groups: int = 6
    n_test_groups: int = 2


@dataclass(frozen=True)
class ExecutionSettings:
    mode: Literal["next_open", "close"] = "next_open"
    initial_capital: float = 1_000_000.0
    periods_per_year: int = 252


@dataclass(frozen=True)
class CostSettings:
    """Tek yön maliyet oranı = komisyon + spread/2 + slippage + k * sqrt(emir / ADV)."""

    commission_bps: float = 5.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    impact_k: float = 0.0                # 0 -> market impact kapalı
    adv_window: int = 20


@dataclass(frozen=True)
class CostScenarioSettings:
    """Parametrik maliyet modeli senaryosu (spread + slippage + market impact).

    Sermaye ızgarası, emir büyüklüğü arttıkça market impact'in nasıl büyüdüğünü
    gösterir. Gerçek ADV yoksa (sentetik veri) sonuç bir SİMÜLASYON VARSAYIMIDIR.
    """

    commission_bps: float = 5.0
    spread_bps: float = 10.0
    slippage_bps: float = 5.0
    impact_k: float = 0.10
    capital_grid: tuple[float, ...] = (1_000_000.0, 10_000_000.0, 100_000_000.0)


@dataclass(frozen=True)
class SizingSettings:
    method: Literal["equal", "vol_target", "prob", "prob_vol", "kelly"] = "prob"
    step_size: float = 0.0
    vol_target_annual: float = 0.15
    kelly_fraction: float = 0.25
    kelly_fractions: tuple[float, ...] = (0.10, 0.25, 0.50)


@dataclass(frozen=True)
class RiskSettings:
    max_position: float = 1.0
    max_gross_exposure: float = 1.0
    max_daily_loss: float | None = None
    max_drawdown_stop: float | None = None
    max_turnover: float | None = None


@dataclass(frozen=True)
class CalibrationSettings:
    methods: tuple[str, ...] = ("platt", "isotonic")
    min_history: int = 100
    n_bins: int = 10


@dataclass(frozen=True)
class BenchmarkSettings:
    random_entry_reps: int = 50


@dataclass(frozen=True)
class RegimeSettings:
    min_history: int = 252
    vol_low_q: float = 0.33
    vol_high_q: float = 0.67
    high_vol_q: float = 0.80
    trend_sma: int = 100
    trend_slope_window: int = 20
    drawdown_window: int = 252
    crisis_drawdown: float = -0.20


@dataclass(frozen=True)
class AnalysisSettings:
    threshold_grid: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    cost_grid_bps: tuple[float, ...] = (5, 10, 20, 30, 50, 75, 100)
    periods: tuple[tuple[str, str, str], ...] = (
        ("2010-2013", "2010-01-01", "2013-12-31"),
        ("2014-2016", "2014-01-01", "2016-12-31"),
        ("2017-2019", "2017-01-01", "2019-12-31"),
        ("2020", "2020-01-01", "2020-12-31"),
        ("2021", "2021-01-01", "2021-12-31"),
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
    )
    feature_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "atr": ("atr_pct", "atr_vol_ratio"),
            "skew": ("ret_skew",),
            "autocorrelation": ("autocorr",),
            "volatility": ("vol_ratio", "bb_width"),
            "volume": ("volume_z",),
            "trend": ("efficiency", "abs_momentum", "rsi"),
        }
    )
    permutation_repeats: int = 5
    bootstrap_n: int = 1000
    bootstrap_block: int = 20
    placebo_label_reps: int = 20
    placebo_feature_reps: int = 20
    placebo_return_reps: int = 20
    placebo_timestamp_reps: int = 20
    leakage_samples: int = 30


@dataclass(frozen=True)
class CriteriaSettings:
    """Robustness testlerinin geçme kriterleri (ex-ante, sonuçlara bakmadan belirlenir)."""

    psr_min: float = 0.95
    dsr_min: float = 0.95
    placebo_alpha: float = 0.05
    random_entry_percentile: float = 0.95
    min_breakeven_cost_bps: float = 20.0
    cpcv_min_positive_frac: float = 0.80
    period_min_positive_frac: float = 0.60


@dataclass(frozen=True)
class ResearchConfig:
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    data: DataSettings = field(default_factory=DataSettings)
    primary: PrimarySettings = field(default_factory=PrimarySettings)
    cusum: CusumSettings = field(default_factory=CusumSettings)
    barriers: BarrierConfig = field(default_factory=BarrierConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    meta_model: MetaModelSettings = field(default_factory=MetaModelSettings)
    cv: CVConfig = field(default_factory=CVConfig)
    cpcv: CPCVSettings = field(default_factory=CPCVSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)
    costs: CostSettings = field(default_factory=CostSettings)
    cost_scenario: CostScenarioSettings = field(default_factory=CostScenarioSettings)
    sizing: SizingSettings = field(default_factory=SizingSettings)
    risk: RiskSettings = field(default_factory=RiskSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    benchmarks: BenchmarkSettings = field(default_factory=BenchmarkSettings)
    regimes: RegimeSettings = field(default_factory=RegimeSettings)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    criteria: CriteriaSettings = field(default_factory=CriteriaSettings)

    # ------------------------------------------------------------ türetilenler
    @property
    def cost_per_side(self) -> float:
        """Birim büyüklükte (market impact hariç) tek yön maliyet oranı."""
        c = self.costs
        return (c.commission_bps + c.spread_bps / 2.0 + c.slippage_bps) / 1e4

    def model_config(self, **overrides: Any) -> MetaModelConfig:
        mm = dataclasses.asdict(self.meta_model)
        cfg = MetaModelConfig(
            **mm,
            step_size=self.sizing.step_size,
            cost_per_side=self.cost_per_side,
            seed=self.experiment.seed,
        )
        return replace(cfg, **overrides) if overrides else cfg

    def primary_model_config(self) -> PrimaryConfig:
        """Birincil model + olay örnekleme ayarlarını çekirdek ``PrimaryConfig``'e birleştirir."""
        return PrimaryConfig(
            **dataclasses.asdict(self.primary),
            event_mode=self.cusum.event_mode,
            cusum_vol_mult=self.cusum.vol_mult,
        )

    def pipeline_config(self) -> PipelineConfig:
        """Çekirdek ``meta_labeling`` pipeline'ının anladığı konfigürasyon."""
        return PipelineConfig(
            sim=self.data.simulation,
            primary=self.primary_model_config(),
            barrier=self.barriers,
            features=self.features,
            cv=self.cv,
            model=self.model_config(),
            periods_per_year=self.execution.periods_per_year,
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


# ---------------------------------------------------------------------- YAML
def _coerce(value: Any, hint: Any, path: str) -> Any:
    if dataclasses.is_dataclass(hint):
        return _build(hint, value, path)
    if isinstance(value, dict):
        return {k: tuple(v) if isinstance(v, list) else v for k, v in value.items()}
    if isinstance(value, list):
        return tuple(tuple(x) if isinstance(x, list) else x for x in value)
    return value


def _build(cls: type, data: dict[str, Any] | None, path: str) -> Any:
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{path}: bilinmeyen anahtar(lar) {sorted(unknown)}")
    hints = typing.get_type_hints(cls)
    kwargs = {k: _coerce(v, hints[k], f"{path}.{k}") for k, v in data.items()}
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def config_from_dict(raw: dict[str, Any]) -> ResearchConfig:
    return _build(ResearchConfig, dict(raw or {}), "config")


def load_research_config(path: str | Path) -> ResearchConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - ortam bağımlı
        raise ImportError("PyYAML gerekli: pip install pyyaml") from exc
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return config_from_dict(raw)
