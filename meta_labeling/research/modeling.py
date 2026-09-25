"""Olay veri seti, walk-forward tahmin, olasılık kalibrasyonu ve pozisyon büyüklüğü.

Çekirdek ``meta_labeling`` fonksiyonları (CUSUM, Triple Barrier, benzersizlik,
PurgedKFold, model fabrikası) yeniden kullanılır; burada yalnızca araştırma
çerçevesinin ihtiyaç duyduğu katman eklenir:

* Etiketler EXECUTION fiyatlarıyla üretilir: ``next_open`` modunda giriş t0+1
  açılışı, çıkış t1+1 açılışıdır. Böylece meta-etiket "gerçekte işlenebilir
  işlem kârlı mıydı?" sorusunu yanıtlar ve sinyal kapanışıyla işlem yapılmaz.
* Purging için etiket bitişi ``label_end`` = çıkış barıdır (t1 değil): etiketin
  bilgisi ancak çıkış fiyatı oluştuğunda tamamlanır.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..cv import PurgedKFold
from ..features import build_feature_matrix
from ..labeling import apply_triple_barrier, get_vertical_barriers
from ..model import build_meta_model, positive_class_proba
from ..primary import build_primary_model
from ..sample_weights import average_uniqueness, num_concurrent_events, return_attribution_weights
from ..sampling import sample_events
from ..sizing import bet_size_from_proba
from ..volatility import get_volatility
from .execution import cost_rate, execution_positions
from .settings import ResearchConfig

log = logging.getLogger("meta_labeling.research")


@dataclass
class MarketState:
    """Bar bazında, olaylardan bağımsız hesaplanan girdiler (bir kez hesaplanır)."""

    ohlcv: pd.DataFrame
    vol: pd.Series
    side: pd.Series
    features: pd.DataFrame
    adv: pd.Series


@dataclass
class EventDataset:
    events: pd.DataFrame      # t0 indeksli
    X: pd.DataFrame
    market: MarketState

    @property
    def y(self) -> pd.Series:
        return self.events["bin"]

    @property
    def label_end(self) -> pd.Series:
        return self.events["label_end"]


def build_market_state(ohlcv: pd.DataFrame, cfg: ResearchConfig, adv: pd.Series) -> MarketState:
    vol = get_volatility(ohlcv["Close"], span=cfg.barriers.vol_span, method=cfg.barriers.vol_method)
    side = build_primary_model(cfg.primary_model_config()).side(ohlcv)
    feats = build_feature_matrix(ohlcv, cfg.features)
    return MarketState(ohlcv=ohlcv, vol=vol, side=side, features=feats, adv=adv)


def sample_candidate_events(market: MarketState, cfg: ResearchConfig) -> pd.DatetimeIndex:
    t_events = sample_events(
        market.side, market.ohlcv["Close"], cfg.cusum.event_mode,
        cusum_threshold=market.vol * cfg.cusum.vol_mult,
    )
    return t_events[(market.vol.reindex(t_events) > cfg.barriers.min_target).to_numpy()]


def build_event_dataset(
    market: MarketState,
    cfg: ResearchConfig,
    t_events: pd.DatetimeIndex | None = None,
    member_mask: pd.Series | None = None,
) -> EventDataset:
    """Olaylar -> Triple Barrier -> execution fiyatlı meta-etiket -> öznitelikler."""
    ohlcv, close = market.ohlcv, market.ohlcv["Close"]
    b = cfg.barriers
    if t_events is None:
        t_events = sample_candidate_events(market, cfg)
    if member_mask is not None:  # survivorship: yalnızca o tarihte endekste olan hisse
        t_events = t_events[member_mask.reindex(t_events).fillna(False).to_numpy(dtype=bool)]
    vertical = get_vertical_barriers(t_events, close.index, b.max_holding_bars)
    ev = apply_triple_barrier(close, t_events, market.vol, market.side, vertical, b.pt_mult, b.sl_mult, b.min_target)

    mode = cfg.execution.mode
    index = ohlcv.index
    if mode == "next_open":  # t1+1 açılışı gerekli
        ev = ev[index.get_indexer(pd.DatetimeIndex(ev["t1"])) + 1 < len(index)]
    entry, exit_ = execution_positions(index, ev.index, ev["t1"], mode)
    px = ohlcv["Open" if mode == "next_open" else "Close"].to_numpy(dtype=float)
    side = ev["side"].to_numpy(dtype=float)
    gross = side * (px[exit_] / px[entry] - 1.0)
    unit = np.ones(len(ev))
    adv = market.adv.to_numpy(dtype=float)
    costs = cost_rate(unit, adv[entry], cfg.costs, cfg.execution.initial_capital) + cost_rate(
        unit, adv[exit_], cfg.costs, cfg.execution.initial_capital
    )
    ev = ev.assign(
        entry_time=index[entry],
        exit_time=index[exit_],
        label_end=index[exit_],
        exec_gross_ret=gross,
        exec_cost=costs,
        exec_net_ret=gross - costs,
    )
    ev["bin"] = (ev["exec_net_ret"] > 0).astype(int)

    X = market.features.reindex(ev.index)
    if cfg.meta_model.use_side_feature:
        X["side"] = ev["side"].astype(float)
    valid = X.notna().all(axis=1)
    ev, X = ev.loc[valid].copy(), X.loc[valid]
    co = num_concurrent_events(index, ev["label_end"])
    ev["uniqueness"] = average_uniqueness(index, ev["label_end"], co)
    ev["return_weight"] = return_attribution_weights(close, ev["label_end"], co)
    return EventDataset(events=ev, X=X, market=market)


# ---------------------------------------------------------------- walk-forward
@dataclass
class OOSPrediction:
    proba: pd.Series                  # OOS P(Y=1); tahmin yoksa NaN
    fold: pd.Series                   # olay -> test fold numarası (-1: tahmin yok)
    models: list = field(default_factory=list)
    splits: list = field(default_factory=list)   # (fold, train_idx, test_idx)


def sample_weight(ds: EventDataset, weighting: str) -> pd.Series:
    if weighting == "uniqueness":
        u = ds.events["uniqueness"]
        return u / u.mean()
    if weighting == "return":
        return ds.events["return_weight"]
    if weighting == "none":
        return pd.Series(1.0, index=ds.events.index)
    raise ValueError(f"Bilinmeyen ağırlıklandırma: {weighting}")


def make_estimator(ds: EventDataset, cfg: ResearchConfig, seed: int | None = None):
    mc = cfg.model_config(seed=cfg.experiment.seed if seed is None else seed)
    return build_meta_model(mc, avg_uniqueness=float(ds.events["uniqueness"].mean()))


def walk_forward(
    ds: EventDataset,
    cfg: ResearchConfig,
    *,
    weighting: str | None = None,
    y: pd.Series | None = None,
    X: pd.DataFrame | None = None,
    seed: int | None = None,
) -> OOSPrediction:
    """Purged walk-forward: her fold yalnızca etiketi test başlangıcından önce KAPANMIŞ olaylarla eğitilir."""
    X = ds.X if X is None else X
    y = ds.y if y is None else y
    w = sample_weight(ds, weighting or cfg.meta_model.weighting).to_numpy()
    est = make_estimator(ds, cfg, seed)
    cv = PurgedKFold(ds.label_end, cfg.cv.n_splits, cfg.cv.embargo_pct, walk_forward=True)
    proba = pd.Series(np.nan, index=X.index, name="proba")
    fold = pd.Series(-1, index=X.index, name="fold")
    out = OOSPrediction(proba=proba, fold=fold)
    for k, (tr, te) in enumerate(cv.split(X)):
        out.splits.append((k, tr, te))
        if len(tr) < cfg.cv.min_train_events or y.iloc[tr].nunique() < 2:
            continue
        model = clone(est).fit(X.iloc[tr], y.iloc[tr], sample_weight=w[tr])
        proba.iloc[te] = positive_class_proba(model, X.iloc[te])
        fold.iloc[te] = k
        out.models.append((k, model))
    return out


# ---------------------------------------------------------------- kalibrasyon
@dataclass
class CalibrationResult:
    proba: pd.Series                               # kalibre olasılık (kalibre edilemeyen fold'lar NaN)
    fit_windows: list[tuple[int, pd.Timestamp, pd.Timestamp, int]]  # (fold, fit_son_etiket, test_başı, n)


def _fit_calibrator(method: str, p: np.ndarray, y: np.ndarray):
    if method == "platt":
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        lr = LogisticRegression(C=1e6).fit(logit.reshape(-1, 1), y)
        return lambda q: lr.predict_proba(
            np.log(np.clip(q, 1e-6, 1 - 1e-6) / np.clip(1 - q, 1e-6, 1)).reshape(-1, 1)
        )[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return iso.predict
    raise ValueError(f"Bilinmeyen kalibrasyon yöntemi: {method}")


def calibrate_walk_forward(
    oos: OOSPrediction, y: pd.Series, label_end: pd.Series, method: str, min_history: int
) -> CalibrationResult:
    """Kalibratör her fold için yalnızca GEÇMİŞ fold'ların OOS tahminleriyle fit edilir.

    Ek koşul: kullanılan geçmiş olayların etiketi, test fold'unun ilk olayından
    ÖNCE kapanmış olmalıdır (label_end < test başlangıcı). Gelecekteki OOS
    verisi kalibrasyon için hiçbir koşulda kullanılmaz.
    """
    calibrated = pd.Series(np.nan, index=oos.proba.index, name=f"proba_{method}")
    windows = []
    folds = sorted(f for f in oos.fold.unique() if f >= 0)
    for f in folds:
        test = oos.fold == f
        test_start = oos.proba.index[test.to_numpy()].min()
        hist = (oos.fold >= 0) & (oos.fold < f) & (label_end < test_start)
        if hist.sum() < min_history or y[hist].nunique() < 2:
            continue
        fn = _fit_calibrator(method, oos.proba[hist].to_numpy(), y[hist].to_numpy())
        calibrated[test] = fn(oos.proba[test].to_numpy())
        windows.append((f, label_end[hist].max(), test_start, int(hist.sum())))
    return CalibrationResult(proba=calibrated, fit_windows=windows)


def calibration_metrics(p: pd.Series, y: pd.Series, n_bins: int) -> tuple[dict[str, float], pd.DataFrame]:
    """Brier, log-loss, ECE ve güvenilirlik (reliability) tablosu."""
    from sklearn.metrics import brier_score_loss, log_loss

    mask = p.notna()
    p, y = p[mask].clip(1e-6, 1 - 1e-6), y[mask]
    bins = np.minimum((p.to_numpy() * n_bins).astype(int), n_bins - 1)
    table = (
        pd.DataFrame({"bin": bins, "p": p.to_numpy(), "y": y.to_numpy()})
        .groupby("bin")
        .agg(n=("y", "size"), mean_pred=("p", "mean"), frac_pos=("y", "mean"))
    )
    table["bin_range"] = [f"[{b / n_bins:.1f}, {(b + 1) / n_bins:.1f})" for b in table.index]
    ece = float((table["n"] / table["n"].sum() * (table["mean_pred"] - table["frac_pos"]).abs()).sum())
    metrics = {
        "n": int(mask.sum()),
        "Brier": float(brier_score_loss(y, p)),
        "Log Loss": float(log_loss(y, p, labels=[0, 1])),
        "ECE": ece,
        "Mean p": float(p.mean()),
        "Base rate": float(y.mean()),
    }
    return metrics, table.reset_index(drop=True)


# ---------------------------------------------------------------- sizing
def position_sizes(
    method: str,
    proba: pd.Series,
    events: pd.DataFrame,
    cfg: ResearchConfig,
    threshold: float | None = None,
    kelly_fraction: float | None = None,
) -> pd.Series:
    """P(Y=1) > eşik olan olaylar için [0, max_position] aralığında büyüklük.

    Yöntemler (hepsi aynı eşik kapısını kullanır, yalnızca büyüklük farklıdır):
      equal      : 1
      vol_target : (hedef yıllık vol / sqrt(252)) / σ_t
      prob       : Prado  m = 2Φ(z) - 1,  z = (p - 0.5) / sqrt(p(1-p))
      prob_vol   : prob x vol_target
      kelly      : fraction x max(0, p - (1-p)/b),  b = pt / sl  (fraksiyonel, sınırlı)
    Hiçbir durumda max_position aşılmaz; kaldıraç yoktur.
    """
    thr = cfg.meta_model.threshold if threshold is None else threshold
    cap = cfg.risk.max_position
    p = proba.reindex(events.index)
    gate = (p > thr).astype(float)
    daily_target = cfg.sizing.vol_target_annual / np.sqrt(cfg.execution.periods_per_year)
    vol_size = (daily_target / events["trgt"]).clip(upper=cap)
    if method == "equal":
        size = gate
    elif method == "vol_target":
        size = gate * vol_size
    elif method == "prob":
        size = bet_size_from_proba(p.fillna(0.0), thr, cfg.sizing.step_size)
    elif method == "prob_vol":
        size = bet_size_from_proba(p.fillna(0.0), thr, cfg.sizing.step_size) * vol_size
    elif method == "kelly":
        frac = cfg.sizing.kelly_fraction if kelly_fraction is None else kelly_fraction
        b = cfg.barriers.pt_mult / cfg.barriers.sl_mult if cfg.barriers.sl_mult > 0 else 1.0
        size = gate * frac * (p - (1.0 - p) / b).clip(lower=0.0)
    else:
        raise ValueError(f"Bilinmeyen sizing yöntemi: {method}")
    return size.fillna(0.0).clip(0.0, cap).rename("size")
