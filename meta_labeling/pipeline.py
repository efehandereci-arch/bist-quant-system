"""Uçtan uca Meta-Labeling pipeline'ı.

Akış
----
1. Veri          : OHLCV (simülasyon ya da CSV)
2. Birincil model: side ∈ {-1, 0, +1}; CUSUM ile olay örnekleme (yüksek recall)
3. Triple Barrier: dinamik volatiliteye ölçekli pt/sl + dikey bariyer
4. Meta-etiket   : Y = 1 (işlem kârlı) / 0 (zarar)
5. Öznitelikler  : yönden bağımsız rejim göstergeleri (+ opsiyonel side)
6. Ağırlıklar    : ortalama benzersizlik (örtüşen etiketler)
7. Doğrulama     : Purged K-Fold (teşhis) + Purged walk-forward (OOS olasılıklar)
8. Karar         : P(Y=1) > eşik filtresi ve olasılıktan bet sizing
9. Değerlendirme : filtre öncesi / sonrası Win Rate ve Sharpe karşılaştırması
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from .backtest import compare_strategies
from .config import PipelineConfig
from .cv import PurgedKFold, cross_validate_purged, walk_forward_predict
from .data import simulate_ohlcv, validate_ohlcv
from .features import build_feature_matrix
from .labeling import apply_triple_barrier, get_meta_labels, get_vertical_barriers
from .model import build_meta_model, feature_importance, positive_class_proba
from .primary import PrimaryModel, build_primary_model
from .sample_weights import average_uniqueness, num_concurrent_events, return_attribution_weights
from .sampling import sample_events
from .sizing import bet_size_from_proba, meta_signals
from .volatility import get_volatility

STRATEGY_LABELS = {
    "primary": "Birincil (filtresiz)",
    "filtered": "Meta filtre",
    "sized": "Meta filtre + bet sizing",
}


@dataclass
class PipelineResult:
    """Pipeline çıktıları. ``summary()`` insan tarafından okunabilir rapor üretir."""

    config: PipelineConfig
    primary_name: str
    ohlcv: pd.DataFrame
    events: pd.DataFrame           # t0 indeksli: t1, trgt, side, barrier, ret, bin, weight, proba, ...
    X: pd.DataFrame                # meta-model öznitelikleri (events ile aynı indeks)
    cv_scores: pd.DataFrame        # Purged K-Fold fold skorları
    comparison: pd.DataFrame       # strateji karşılaştırması (OOS)
    oos_metrics: dict[str, float]
    importance: pd.Series
    final_model: object            # tüm etiketli olaylarla eğitilmiş, canlı kullanım modeli
    extras: dict = field(default_factory=dict)

    def summary(self) -> str:
        return format_summary(self)


class MetaLabelingPipeline:
    """Meta-Labeling + Triple Barrier araştırma/üretim pipeline'ı.

    Örnek:
        >>> from meta_labeling import MetaLabelingPipeline
        >>> result = MetaLabelingPipeline().run()     # sentetik veri
        >>> print(result.summary())
        >>> live = MetaLabelingPipeline().score_events(ohlcv, result.final_model)
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    # ------------------------------------------------------------------ adımlar
    def build_primary(self) -> PrimaryModel:
        return build_primary_model(self.config.primary)

    def volatility(self, close: pd.Series) -> pd.Series:
        b = self.config.barrier
        return get_volatility(close, span=b.vol_span, method=b.vol_method)

    def candidate_events(self, ohlcv: pd.DataFrame) -> tuple[pd.DatetimeIndex, pd.Series, pd.Series]:
        """Birincil sinyal + olay örnekleme. Etiketi henüz bilinmeyen güncel olayları da içerir.

        Volatilitesi ``min_target``'ın altında kalan olaylar burada elenir; böylece
        eğitim (``label``) ve canlı skorlama (``score_events``) aynı olay kümesini görür.
        """
        close = ohlcv["Close"]
        vol = self.volatility(close)
        side = self.build_primary().side(ohlcv)
        p = self.config.primary
        t_events = sample_events(side, close, p.event_mode, cusum_threshold=vol * p.cusum_vol_mult)
        t_events = t_events[(vol.reindex(t_events) > self.config.barrier.min_target).to_numpy()]
        return t_events, side, vol

    def label(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Adım 2-4: olaylar, Triple Barrier ve meta-etiketler."""
        close = ohlcv["Close"]
        b = self.config.barrier
        t_events, side, vol = self.candidate_events(ohlcv)
        vertical = get_vertical_barriers(t_events, close.index, b.max_holding_bars)
        events = apply_triple_barrier(
            close, t_events, vol, side, vertical, b.pt_mult, b.sl_mult, b.min_target
        )
        return get_meta_labels(events, self.config.model.cost_per_side)

    def features(self, ohlcv: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """Adım 5: olay anındaki (t0 kapanışı) öznitelikler."""
        X = build_feature_matrix(ohlcv, self.config.features).reindex(events.index)
        if self.config.model.use_side_feature:
            X["side"] = events["side"].astype(float)
        return X

    def sample_weights(self, close: pd.Series, events: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Adım 6: (eğitim ağırlıkları, ortalama benzersizlik)."""
        co = num_concurrent_events(close.index, events["t1"])
        uniq = average_uniqueness(close.index, events["t1"], co)
        scheme = self.config.model.weighting
        if scheme == "uniqueness":
            w = uniq / uniq.mean()
        elif scheme == "return":
            w = return_attribution_weights(close, events["t1"], co)
        elif scheme == "none":
            w = pd.Series(1.0, index=events.index)
        else:
            raise ValueError(f"Bilinmeyen ağırlıklandırma: {scheme}")
        return w.rename("weight"), uniq

    # ------------------------------------------------------------------ çalıştır
    def run(self, ohlcv: pd.DataFrame | None = None) -> PipelineResult:
        cfg = self.config
        ohlcv = validate_ohlcv(ohlcv) if ohlcv is not None else simulate_ohlcv(cfg.sim)
        close = ohlcv["Close"]

        # 1-4) Etiketleme
        events = self.label(ohlcv)

        # 5) Öznitelikler; ısınma dönemindeki eksik gözlemler atılır
        X = self.features(ohlcv, events)
        valid = X.notna().all(axis=1)
        events, X = events.loc[valid].copy(), X.loc[valid]
        y = events["bin"]
        if len(events) < 2 * cfg.cv.n_splits or y.nunique() < 2:
            raise ValueError("Meta-model eğitimi için yeterli/çeşitli etiket yok")

        # 6) Örnek ağırlıkları
        w, uniq = self.sample_weights(close, events)
        events["weight"], events["uniqueness"] = w, uniq
        estimator = build_meta_model(cfg.model, avg_uniqueness=float(uniq.mean()))

        # 7a) Purged K-Fold teşhisleri
        cv_kfold = PurgedKFold(events["t1"], cfg.cv.n_splits, cfg.cv.embargo_pct)
        cv_scores = cross_validate_purged(estimator, X, y, cv_kfold, w, cfg.model.threshold)

        # 7b) Purged walk-forward OOS olasılıkları (backtest girdisi)
        cv_wf = PurgedKFold(events["t1"], cfg.cv.n_splits, cfg.cv.embargo_pct, walk_forward=True)
        proba, wf_models = walk_forward_predict(
            estimator, X, y, cv_wf, w, min_train_events=cfg.cv.min_train_events
        )
        events["proba"] = proba
        oos = events.loc[proba.notna()]
        if oos.empty:
            raise ValueError("OOS tahmin üretilemedi; daha fazla veri ya da daha düşük min_train_events gerekli")

        # 8) Sinyal filtreleme & bet sizing
        signals = meta_signals(oos["side"], oos["proba"], cfg.model.threshold, cfg.model.step_size)
        events["size"] = bet_size_from_proba(oos["proba"], cfg.model.threshold, cfg.model.step_size)

        # 9) Karşılaştırma (aynı OOS olayları, aynı takvim penceresi)
        comparison = compare_strategies(
            close, oos["t1"], signals, cfg.model.cost_per_side, cfg.periods_per_year
        )
        comparison.index = [
            f"{STRATEGY_LABELS[k]} (p>{cfg.model.threshold:.2f})" if k != "primary" else STRATEGY_LABELS[k]
            for k in comparison.index
        ]

        taken = oos["proba"] > cfg.model.threshold
        y_oos = oos["bin"]
        oos_metrics = {
            "n_oos": float(len(oos)),
            "auc": float(roc_auc_score(y_oos, oos["proba"])) if y_oos.nunique() == 2 else float("nan"),
            "base_precision": float(y_oos.mean()),
            "meta_precision": float(y_oos[taken].mean()) if taken.any() else float("nan"),
            "meta_recall": float(y_oos[taken].sum() / max(y_oos.sum(), 1)),
            "coverage": float(taken.mean()),
        }

        # Canlı kullanım için tüm etiketli veriyle eğitilmiş nihai model
        final_model = clone(estimator).fit(X, y, sample_weight=w.to_numpy())

        return PipelineResult(
            config=cfg,
            primary_name=self.build_primary().name,
            ohlcv=ohlcv,
            events=events,
            X=X,
            cv_scores=cv_scores,
            comparison=comparison,
            oos_metrics=oos_metrics,
            importance=feature_importance(wf_models, X.columns),
            final_model=final_model,
        )

    # ------------------------------------------------------------------ canlı
    def score_events(self, ohlcv: pd.DataFrame, model, last_n: int | None = None) -> pd.DataFrame:
        """Güncel veride olayları üretir ve eğitilmiş meta-modelle puanlar.

        Etiketi henüz oluşmamış (dikey bariyeri gelecekte olan) son olaylar da
        dahildir; canlı işlem kararı bunların sonuncusundan okunur.
        """
        ohlcv = validate_ohlcv(ohlcv)
        t_events, side, _ = self.candidate_events(ohlcv)
        frame = pd.DataFrame({"side": side.reindex(t_events)}, index=t_events)
        X = self.features(ohlcv, frame).dropna()
        if last_n is not None:
            X = X.iloc[-last_n:]
        proba = pd.Series(positive_class_proba(model, X), index=X.index, name="proba")
        m = self.config.model
        size = bet_size_from_proba(proba, m.threshold, m.step_size)
        return pd.DataFrame(
            {"side": frame["side"].reindex(X.index), "proba": proba, "size": size,
             "signal": frame["side"].reindex(X.index) * size}
        )


# ---------------------------------------------------------------------- rapor
def _pct(x: float) -> str:
    return "  n/a" if pd.isna(x) else f"{100 * x:5.1f}%"


def format_summary(res: PipelineResult) -> str:
    cfg, ev, cmp_ = res.config, res.events, res.comparison
    b, m, cvc = cfg.barrier, cfg.model, cfg.cv
    lines: list[str] = []
    bar = "=" * 78
    lines += [bar, " META-LABELING + TRIPLE BARRIER PIPELINE — ÖZET", bar]
    idx = res.ohlcv.index
    lines.append(
        f"Veri          : {len(idx)} bar ({idx[0].date()} → {idx[-1].date()})"
    )
    lines.append(
        f"Birincil model: {res.primary_name}, olay örnekleme = {cfg.primary.event_mode}"
    )
    lines.append(
        f"Bariyerler    : pt={b.pt_mult}σ, sl={b.sl_mult}σ, dikey={b.max_holding_bars} bar, "
        f"σ = {b.vol_method.upper()}({b.vol_span}) log-getiri volatilitesi"
    )
    dist = ev["barrier"].value_counts(normalize=True)
    lines.append(
        f"Olaylar       : {len(ev)} | Y=1 oranı {_pct(ev['bin'].mean())} | "
        f"ort. benzersizlik {ev['uniqueness'].mean():.2f} | "
        f"temas: pt {_pct(dist.get('pt', 0))}, sl {_pct(dist.get('sl', 0))}, "
        f"dikey {_pct(dist.get('vertical', 0))}"
    )

    lines += ["", f"[1] Purged K-Fold CV (k={cvc.n_splits}, embargo={cvc.embargo_pct:.0%}) — meta-model teşhisi"]
    cols = ["n_train", "n_test", "auc", "log_loss", "precision", "recall", "base_rate"]
    cv_tbl = res.cv_scores[cols].copy()
    lines.append(cv_tbl.to_string(float_format=lambda v: f"{v:.3f}"))
    lines.append(
        f"  ortalama AUC = {cv_tbl['auc'].mean():.3f} ± {cv_tbl['auc'].std():.3f} | "
        f"precision {cv_tbl['precision'].mean():.3f} vs baz oran {cv_tbl['base_rate'].mean():.3f}"
    )

    om = res.oos_metrics
    lines += [
        "",
        "[2] Purged walk-forward OOS (her fold yalnızca geçmişte kapanmış etiketlerle eğitilir)",
        f"  OOS olay: {int(om['n_oos'])} | AUC: {om['auc']:.3f} | "
        f"birincil precision (baz): {_pct(om['base_precision'])} → meta precision: "
        f"{_pct(om['meta_precision'])} | recall: {_pct(om['meta_recall'])} | "
        f"kapsama: {_pct(om['coverage'])}",
    ]

    lines += ["", f"[3] Strateji karşılaştırması (OOS, maliyet {m.cost_per_side * 1e4:.0f} bps/yön)"]
    view = pd.DataFrame(
        {
            "İşlem": cmp_["n_trades"].astype(int),
            "WinRate": cmp_["win_rate"].map(_pct),
            "Ort.İşlem": cmp_["avg_trade_ret"].map(lambda v: f"{100 * v:+.2f}%"),
            "İşlemSR": cmp_["trade_sharpe"].map(lambda v: f"{v:5.2f}"),
            "Sharpe": cmp_["sharpe"].map(lambda v: f"{v:5.2f}"),
            "PSR": cmp_["psr"].map(_pct),
            "Yıl.Getiri": cmp_["ann_return"].map(_pct),
            "MaxDD": cmp_["max_drawdown"].map(_pct),
        }
    )
    lines.append(view.to_string())
    base, filt = cmp_.iloc[0], cmp_.iloc[1]
    lines.append(
        f"  Δ Win Rate (filtre - birincil): {100 * (filt['win_rate'] - base['win_rate']):+.1f} puan | "
        f"Δ Sharpe: {filt['sharpe'] - base['sharpe']:+.2f} | "
        f"işlem sayısı {int(base['n_trades'])} → {int(filt['n_trades'])}"
    )

    if not res.importance.empty:
        top = ", ".join(f"{k} {v:.2f}" for k, v in res.importance.head(5).items())
        lines += ["", f"[4] Öznitelik önemi (MDI, walk-forward ort., yalnızca araştırma amaçlı): {top}"]
    lines += [
        "",
        "Not: Eşik ve hiperparametreler ex-ante sabittir; OOS sonuçlara bakarak ayarlamak",
        "     backtest overfitting'e yol açar (AFML Bölüm 11). Sharpe günlük portföy getirileri",
        "     üzerinden, PSR ise P(SR > 0) olarak raporlanmıştır.",
        bar,
    ]
    return "\n".join(lines)
