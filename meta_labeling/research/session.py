"""Araştırma oturumu: notebook hücrelerine birebir karşılık gelen aşamalar.

Her aşama sonucu ``self.results[<aşama>]`` altında saklanır ve bir kez
hesaplanır (tekrarlanan hesaplama yok). ``run_all()`` tüm aşamaları sırayla
çalıştırır. Hiçbir aşama OOS sonuçlarına bakarak parametre seçmez: eşik,
hiperparametreler, bariyerler ve öznitelikler config'den gelir; eşik ızgarası,
maliyet ızgarası, ablation ve sizing karşılaştırmaları YALNIZCA raporlanır ve
Deflated Sharpe hesabında "deneme" (trial) olarak sayılır.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from ..cv import PurgedKFold, cross_validate_purged
from ..labeling import apply_triple_barrier, get_vertical_barriers
from ..model import positive_class_proba
from .cpcv import CombinatorialPurgedKFold
from .execution import (
    PortfolioResult,
    adv_value,
    events_to_target,
    segment_trades,
    simulate_portfolio,
    trade_ledger,
)
from .importance import mdi_importance, permutation_importance_oos, shap_importance_oos
from .leakage import LeakageReport, run_leakage_audit
from .metrics import (
    bootstrap_returns,
    bootstrap_trades,
    deflated_sharpe,
    drawdown_summary,
    per_period_sharpe,
    performance_metrics,
    trade_distribution,
)
from .modeling import (
    EventDataset,
    MarketState,
    OOSPrediction,
    build_event_dataset,
    build_market_state,
    calibrate_walk_forward,
    calibration_metrics,
    make_estimator,
    position_sizes,
    sample_candidate_events,
    sample_weight,
    walk_forward,
)
from .regimes import UNKNOWN, classify_regimes
from .settings import CostSettings, ResearchConfig, load_research_config
from .universe import SurvivorshipControl, load_market, validate_market_data

MAIN = ("Primary", "Meta", "Meta+Sizing", "Buy&Hold")


@dataclass
class StrategyRun:
    name: str
    port: PortfolioResult
    trades: pd.Series
    metrics: dict[str, float]
    ledger: pd.DataFrame | None = None


def safe_auc(y: pd.Series, p: pd.Series) -> float:
    mask = p.notna()
    y, p = y[mask], p[mask]
    return float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan")


def setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("meta_labeling.research")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class ResearchSession:
    """Tek bir hisse için uçtan uca araştırma/backtest oturumu."""

    def __init__(self, cfg: ResearchConfig, ticker: str | None = None):
        self.cfg = cfg
        self.ticker = ticker or cfg.data.tickers[0]
        self.log = setup_logging(cfg.experiment.log_level)
        self.out_dir = Path(cfg.experiment.output_dir) / self.ticker
        self.results: dict[str, Any] = {}
        self.trials: dict[str, float] = {}       # DSR için: varyant -> periyot Sharpe'ı
        self.leakage: dict[str, LeakageReport] = {}
        self._runs: dict[str, StrategyRun] = {}
        self.ohlcv: pd.DataFrame | None = None
        self.market: MarketState | None = None
        self.dataset: EventDataset | None = None
        self.oos: OOSPrediction | None = None

    @classmethod
    def from_yaml(cls, path: str | Path, ticker: str | None = None) -> "ResearchSession":
        return cls(load_research_config(path), ticker)

    # ----------------------------------------------------------- yardımcılar
    def rng(self, offset: int = 0) -> np.random.Generator:
        return np.random.default_rng(self.cfg.experiment.seed + offset)

    def _require(self, attr: str, stage: str) -> Any:
        value = getattr(self, attr)
        if value is None:
            raise RuntimeError(f"Önce '{stage}' aşamasını çalıştırın")
        return value

    @property
    def ppy(self) -> int:
        return self.cfg.execution.periods_per_year

    def run_signal(
        self,
        name: str,
        signal: pd.Series,
        t1: pd.Series,
        *,
        window: tuple[pd.Timestamp, pd.Timestamp] | None = None,
        ohlcv: pd.DataFrame | None = None,
        adv: pd.Series | None = None,
        costs: CostSettings | None = None,
        capital: float | None = None,
        register: bool = False,
    ) -> StrategyRun:
        """Olay sinyalini (side x size) execution/maliyet/risk modeliyle simüle eder."""
        cfg = self.cfg
        ohlcv = self.ohlcv if ohlcv is None else ohlcv
        adv = self.market.adv if adv is None else adv
        start, end = window or self.window
        win = ohlcv.loc[start:end]
        execu = cfg.execution if capital is None else replace(cfg.execution, initial_capital=capital)
        costs = costs or cfg.costs
        sig = signal[(signal != 0) & (signal.index >= start)].astype(float)
        target = (
            events_to_target(win.index, sig.index, t1, sig) if len(sig) else pd.Series(0.0, index=win.index)
        )
        port = simulate_portfolio(target, win, execu, costs, cfg.risk, adv)
        ledger = trade_ledger(win, t1, sig, execu, costs, adv)
        run = StrategyRun(name, port, ledger["net"], performance_metrics(port, ledger["net"], self.ppy), ledger)
        if register:
            self.trials[name] = per_period_sharpe(port.returns)
        return run

    def run_target(self, name: str, target: pd.Series, window=None) -> StrategyRun:
        """Sürekli pozisyonlu stratejiler (buy & hold, sürekli EMA kuralı)."""
        start, end = window or self.window
        win = self.ohlcv.loc[start:end]
        port = simulate_portfolio(target.reindex(win.index).fillna(0.0), win, self.cfg.execution,
                                  self.cfg.costs, self.cfg.risk, self.market.adv)
        trades = segment_trades(port)
        return StrategyRun(name, port, trades, performance_metrics(port, trades, self.ppy))

    @property
    def oos_events(self) -> pd.DataFrame:
        oos = self._require("oos", "12_walk_forward")
        ev = self.dataset.events.loc[oos.proba.notna()].copy()
        ev["proba"] = oos.proba[oos.proba.notna()]
        return ev

    @property
    def window(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        ev = self.oos_events
        return ev.index.min(), ev["exit_time"].max()

    def meta_signal(self, proba: pd.Series, events: pd.DataFrame | None = None,
                    method: str = "equal", threshold: float | None = None, **kw) -> pd.Series:
        events = self.oos_events if events is None else events
        size = position_sizes(method, proba, events, self.cfg, threshold=threshold, **kw)
        return (events["side"] * size).rename("signal")

    # ================================================================ 03 / 04
    def load_data(self) -> dict[str, Any]:
        if self.ohlcv is None:
            md = load_market(self.ticker, self.cfg.data)
            self.market_data, self.ohlcv = md, md.ohlcv
            self.survivorship = SurvivorshipControl(self.cfg.data.constituents_file)
            self.member_mask = self.survivorship.membership(self.ticker, self.ohlcv.index)
            idx = self.ohlcv.index
            self.results["data"] = {
                "ticker": self.ticker,
                "source": md.source,
                "simulated": md.simulated,
                "bars": len(idx),
                "start": idx[0].date(),
                "end": idx[-1].date(),
                "notes": md.notes,
                "survivorship": self.survivorship.limitation(len(self.cfg.data.tickers)),
            }
            self.log.info("Veri yüklendi: %s, %d bar (%s → %s)", self.ticker, len(idx), idx[0].date(), idx[-1].date())
        return self.results["data"]

    def validate_data(self) -> pd.DataFrame:
        self.load_data()
        if "validation" not in self.results:
            self.results["validation"] = validate_market_data(self.ohlcv)
        return self.results["validation"]

    # ================================================================ 05
    def build_features(self) -> pd.DataFrame:
        self.load_data()
        if self.market is None:
            adv = adv_value(self.ohlcv, self.cfg.costs.adv_window)
            self.market = build_market_state(self.ohlcv, self.cfg, adv)
            self.regimes = classify_regimes(self.ohlcv, self.market.vol, self.cfg.regimes)
            self.results["features"] = self.market.features.describe().T[["mean", "std", "min", "max"]]
        return self.results["features"]

    # ================================================================ 06
    def leakage_audit(self, full: bool = False) -> LeakageReport:
        """``full=False``: öznitelik/volatilite/sinyal/rejim kesme testleri.
        ``full=True``: + etiket sırası, CV bölmeleri, CPCV, kalibrasyon kontrolleri."""
        self.build_features()
        splits, calib = {}, None
        if full:
            ds = self._require("dataset", "08_triple_barrier")
            if self.oos is not None:
                splits["walk_forward"] = (ds.label_end, [(tr, te) for _, tr, te in self.oos.splits], True)
            if "kfold_splits" in self.results:
                splits["purged_kfold"] = (ds.label_end, self.results["kfold_splits"], False)
            if "cpcv_splits" in self.results:
                splits["cpcv"] = (ds.label_end, self.results["cpcv_splits"], False)
            calib = self.results.get("calibration", {}).get("fit_windows")
        report = run_leakage_audit(self.ohlcv, self.cfg, dataset=self.dataset if full else None,
                                   splits=splits, calibration_windows=calib)
        self.leakage["full" if full else "features"] = report
        return report

    # ================================================================ 07
    def sample_events(self) -> dict[str, Any]:
        self.build_features()
        if "events" not in self.results:
            self.t_events = sample_candidate_events(self.market, self.cfg)
            side = self.market.side.reindex(self.t_events)
            self.results["events"] = {
                "mode": self.cfg.cusum.event_mode,
                "cusum_threshold": f"{self.cfg.cusum.vol_mult} x σ_t",
                "n_events": len(self.t_events),
                "long": int((side > 0).sum()),
                "short": int((side < 0).sum()),
                "events_per_100_bars": 100 * len(self.t_events) / len(self.ohlcv),
            }
        return self.results["events"]

    # ================================================================ 08
    def triple_barrier(self) -> pd.DataFrame:
        self.sample_events()
        if self.dataset is None:
            self.dataset = build_event_dataset(self.market, self.cfg, self.t_events, self.member_mask)
            ev = self.dataset.events
            index = self.ohlcv.index
            hold = index.get_indexer(pd.DatetimeIndex(ev["t1"])) - index.get_indexer(ev.index)
            self.results["barrier"] = {
                "n": len(ev),
                "touch": ev["barrier"].value_counts(normalize=True).to_dict(),
                "avg_holding_bars": float(hold.mean()),
                "avg_uniqueness": float(ev["uniqueness"].mean()),
                "avg_target_sigma": float(ev["trgt"].mean()),
            }
        return self.dataset.events

    # ================================================================ 09
    def primary_model(self) -> pd.DataFrame:
        ev = self.triple_barrier()
        if "primary" not in self.results:
            rows = {}
            for name, sub in (("All", ev), ("Long", ev[ev["side"] > 0]), ("Short", ev[ev["side"] < 0])):
                rows[name] = {"events": len(sub), "precision (Y=1 rate)": sub["bin"].mean(),
                              "avg net ret": sub["exec_net_ret"].mean()}
            self.results["primary"] = pd.DataFrame(rows).T
        return self.results["primary"]

    # ================================================================ 10
    def meta_labels(self) -> dict[str, Any]:
        ev = self.triple_barrier()
        self.results["labels"] = {
            "definition": "Y=1 <=> side*(P_exit/P_entry - 1) - (giriş+çıkış maliyeti) > 0, "
                          f"execution={self.cfg.execution.mode}",
            "n": len(ev),
            "y1_rate": float(ev["bin"].mean()),
            "uniqueness": ev["uniqueness"].describe().to_dict(),
        }
        return self.results["labels"]

    # ================================================================ 11
    def cross_validation(self) -> pd.DataFrame:
        ds = self._require("dataset", "08_triple_barrier")
        if "cv" not in self.results:
            cv = PurgedKFold(ds.label_end, self.cfg.cv.n_splits, self.cfg.cv.embargo_pct)
            self.results["kfold_splits"] = list(cv.split(ds.X))
            w = sample_weight(ds, self.cfg.meta_model.weighting)
            self.results["cv"] = cross_validate_purged(
                make_estimator(ds, self.cfg), ds.X, ds.y, cv, w, self.cfg.meta_model.threshold
            )
        return self.results["cv"]

    # ================================================================ 12
    def walk_forward(self) -> pd.DataFrame:
        ds = self._require("dataset", "08_triple_barrier")
        if self.oos is None:
            main_w = self.cfg.meta_model.weighting
            runs = {main_w: walk_forward(ds, self.cfg)}
            for w in ("none", "uniqueness"):
                if w not in runs:
                    runs[w] = walk_forward(ds, self.cfg, weighting=w)
            self.oos = runs[main_w]
            ev = self.oos_events
            rows = {}
            for w, oos in runs.items():
                p = oos.proba.reindex(ev.index)
                run = self.run_signal(f"weighting={w}", self.meta_signal(p), ev["t1"], register=True)
                taken = p > self.cfg.meta_model.threshold
                rows[f"{'A' if w == 'none' else 'B'}) {w}"] = {
                    "AUC": safe_auc(ev["bin"], p),
                    "Precision": ev["bin"][taken].mean() if taken.any() else np.nan,
                    "Coverage": taken.mean(),
                    **{k: run.metrics[k] for k in ("Trades", "Win Rate", "Sharpe", "Max Drawdown", "CAGR")},
                }
            self.results["weighting"] = pd.DataFrame(rows).T.sort_index()
            self.results["oos"] = {
                "n_oos": len(ev),
                "window": self.window,
                "train_start": self.dataset.events.index.min(),
                "auc": safe_auc(ev["bin"], ev["proba"]),
                "base_precision": float(ev["bin"].mean()),
            }
            self.log.info("Walk-forward OOS: %d olay, AUC=%.3f", len(ev), self.results["oos"]["auc"])
        return self.results["weighting"]

    # ================================================================ 13
    def calibration(self) -> pd.DataFrame:
        if "calibration" in self.results:
            return self.results["calibration"]["metrics"]
        ev = self.oos_events
        ds = self.dataset
        cc = self.cfg.calibration
        probs = {"raw": self.oos.proba}
        windows = {}
        for m in cc.methods:
            res = calibrate_walk_forward(self.oos, ds.y, ds.label_end, m, cc.min_history)
            probs[m], windows[m] = res.proba, res.fit_windows
        common = ev.index[np.all([probs[k].reindex(ev.index).notna() for k in probs], axis=0)]
        sub = ev.loc[common]
        metrics, reliability = {}, {}
        win = (sub.index.min(), sub["exit_time"].max()) if len(sub) else self.window
        for k, p in probs.items():
            pk = p.reindex(common)
            m_, rel = calibration_metrics(pk, sub["bin"], cc.n_bins)
            run = self.run_signal(f"calibration={k}", self.meta_signal(pk, sub), sub["t1"], window=win, register=True)
            m_.update({"Trades": run.metrics["Trades"], "Win Rate": run.metrics["Win Rate"],
                       "Sharpe": run.metrics["Sharpe"], "AUC": safe_auc(sub["bin"], pk)})
            metrics[k], reliability[k] = m_, rel
        self.results["calibration"] = {
            "metrics": pd.DataFrame(metrics).T,
            "reliability": reliability,
            "probs": {k: v.reindex(common) for k, v in probs.items()},
            "fit_windows": windows,
            "n_common": len(common),
            "calibrated_proba": probs.get("isotonic", probs["raw"]),
        }
        return self.results["calibration"]["metrics"]

    # ================================================================ 14
    def randomized_label_runs(self) -> list[dict[str, float]]:
        """Placebo A / benchmark F: etiketler karıştırılarak yeniden eğitilen meta-model."""
        if "randomized_labels" not in self.results:
            ds, ev = self.dataset, self.oos_events
            out = []
            for r in range(self.cfg.analysis.placebo_label_reps):
                y_perm = pd.Series(self.rng(1000 + r).permutation(ds.y.to_numpy()), index=ds.y.index)
                oos = walk_forward(ds, self.cfg, y=y_perm, seed=self.cfg.experiment.seed + r)
                p = oos.proba.reindex(ev.index)
                run = self.run_signal(f"randlabel_{r}", self.meta_signal(p), ev["t1"])
                out.append({**run.metrics, "AUC": safe_auc(ev["bin"], p)})
            self.results["randomized_labels"] = out
        return self.results["randomized_labels"]

    def random_entry_runs(self) -> list[dict[str, float]]:
        if "random_entry" not in self.results:
            ev, cfg = self.oos_events, self.cfg
            index = self.ohlcv.index
            start, end = self.window
            lo = index.get_indexer([start])[0]
            hi = index.get_indexer([end])[0] - cfg.barriers.max_holding_bars - 2
            vol = self.market.vol
            out = []
            for r in range(cfg.benchmarks.random_entry_reps):
                rng = self.rng(2000 + r)
                pos = np.sort(rng.choice(np.arange(lo, hi), size=min(len(ev), hi - lo), replace=False))
                t_ev = index[pos]
                side = pd.Series(rng.choice([-1, 1], size=len(t_ev)), index=t_ev)
                vb = get_vertical_barriers(t_ev, index, cfg.barriers.max_holding_bars)
                rev = apply_triple_barrier(self.ohlcv["Close"], t_ev, vol, side, vb,
                                           cfg.barriers.pt_mult, cfg.barriers.sl_mult, 0.0)
                run = self.run_signal(f"random_{r}", rev["side"].astype(float), rev["t1"])
                out.append(run.metrics)
            self.results["random_entry"] = out
        return self.results["random_entry"]

    def benchmarks(self) -> pd.DataFrame:
        if "benchmarks" in self.results:
            return self.results["benchmarks"]
        cfg, ev = self.cfg, self.oos_events
        thr = cfg.meta_model.threshold
        runs: dict[str, StrategyRun] = {}
        runs["A) Buy & Hold"] = self.run_target("Buy&Hold", pd.Series(1.0, index=self.ohlcv.index))
        runs["B) Primary EMA rule (always in market)"] = self.run_target("EMA rule", self.market.side.astype(float))
        runs["C) Primary + equal sizing"] = self.run_signal("Primary", ev["side"].astype(float), ev["t1"])
        vol_size = position_sizes("vol_target", pd.Series(1.0, index=ev.index), ev, cfg, threshold=0.0)
        runs["D) Primary + volatility sizing"] = self.run_signal("Primary vol", ev["side"] * vol_size, ev["t1"])
        runs["H) Meta model"] = self.run_signal("Meta", self.meta_signal(ev["proba"]), ev["t1"], register=True)
        runs["I) Meta model + bet sizing"] = self.run_signal(
            "Meta+Sizing", self.meta_signal(ev["proba"], method=cfg.sizing.method), ev["t1"], register=True
        )
        # G) Majority-class: her fold'da eğitim baz oranını sabit olasılık olarak tahmin eder
        base = pd.Series(np.nan, index=self.dataset.events.index)
        for k, tr, te in self.oos.splits:
            if (self.oos.fold.iloc[te] == k).any():
                base.iloc[te] = self.dataset.y.iloc[tr].mean()
        runs["G) Majority-class predictor"] = self.run_signal(
            "Majority", self.meta_signal(base.reindex(ev.index)), ev["t1"]
        )
        table = {k: r.metrics for k, r in runs.items()}
        rand = pd.DataFrame(self.random_entry_runs())
        table["E) Random entry (median)"] = rand.median(numeric_only=True).to_dict()
        rl = pd.DataFrame(self.randomized_label_runs())
        table["F) Randomized-label meta (median)"] = rl.drop(columns="AUC").median(numeric_only=True).to_dict()
        order = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        df = pd.DataFrame(table).T
        df = df.loc[sorted(df.index, key=lambda s: order.index(s[0]))]
        self._bench_runs = runs
        self._runs.update({
            "Primary": runs["C) Primary + equal sizing"],
            "Meta": runs["H) Meta model"],
            "Meta+Sizing": runs["I) Meta model + bet sizing"],
            "Buy&Hold": runs["A) Buy & Hold"],
        })
        self.results["benchmarks"] = df
        self.results["benchmark_note"] = (
            f"Tüm stratejiler aynı OOS penceresi ({self.window[0].date()} → {self.window[1].date()}), aynı "
            f"maliyet ({cfg.cost_per_side * 1e4:.1f} bps/yön + impact k={cfg.costs.impact_k}), aynı başlangıç "
            f"sermayesi ({cfg.execution.initial_capital:,.0f}) ve aynı execution ({cfg.execution.mode}) ile test "
            f"edildi. Meta eşiği p>{thr}. E ve F medyan değerlerdir."
        )
        return df

    # ================================================================ 15
    def transaction_costs(self) -> pd.DataFrame:
        if "cost_sensitivity" in self.results:
            return self.results["cost_sensitivity"]
        cfg, ev = self.cfg, self.oos_events
        signals = {
            "Primary": ev["side"].astype(float),
            "Meta": self.meta_signal(ev["proba"]),
            "Meta+Sizing": self.meta_signal(ev["proba"], method=cfg.sizing.method),
        }
        rows = []
        for bps in cfg.analysis.cost_grid_bps:
            costs = CostSettings(commission_bps=float(bps), adv_window=cfg.costs.adv_window)
            for name, sig in signals.items():
                m = self.run_signal(name, sig, ev["t1"], costs=costs).metrics
                rows.append({"cost_bps": bps, "strategy": name, "CAGR": m["CAGR"], "Sharpe": m["Sharpe"],
                             "Max Drawdown": m["Max Drawdown"], "Win Rate": m["Win Rate"],
                             "Turnover": m["Turnover"], "Net Return": m["Total Return"]})
        table = pd.DataFrame(rows)
        self.results["cost_sensitivity"] = table
        meta = table[table["strategy"] == "Meta"].set_index("cost_bps")["Sharpe"]
        self.results["breakeven_cost_bps"] = _breakeven(meta)

        # Parametrik maliyet modeli: sermaye büyüdükçe market impact
        scen = cfg.cost_scenario
        model_costs = CostSettings(commission_bps=scen.commission_bps, spread_bps=scen.spread_bps,
                                   slippage_bps=scen.slippage_bps, impact_k=scen.impact_k,
                                   adv_window=cfg.costs.adv_window)
        impact_rows = []
        for capital in scen.capital_grid:
            for name in ("Meta", "Meta+Sizing"):
                run = self.run_signal(name, signals[name], ev["t1"], costs=model_costs, capital=capital)
                impact_rows.append({"capital": capital, "strategy": name, "Sharpe": run.metrics["Sharpe"],
                                    "CAGR": run.metrics["CAGR"],
                                    "avg cost per trade (bps)": 1e4 * run.ledger["cost"].mean()
                                    if len(run.ledger) else np.nan})
        self.results["cost_model"] = {
            "formula": "rate = commission + spread/2 + slippage + k*sqrt(order_value/ADV)",
            "params": scen,
            "adv_simulated": self.market_data.simulated,
            "table": pd.DataFrame(impact_rows),
        }
        return table

    # ================================================================ 16
    def position_sizing(self) -> pd.DataFrame:
        if "sizing" in self.results:
            return self.results["sizing"]
        cfg, ev = self.cfg, self.oos_events
        calibrated = self.results.get("calibration", {}).get("calibrated_proba")
        p_kelly = ev["proba"]
        n_fallback = len(ev)
        if calibrated is not None:
            cp = calibrated.reindex(ev.index)
            n_fallback = int(cp.isna().sum())
            p_kelly = cp.fillna(ev["proba"])
        rows = {}
        variants = [("A) Equal weight", "equal", {}), ("B) Volatility targeting", "vol_target", {}),
                    ("C) Probability (Prado)", "prob", {}), ("D) Probability x volatility", "prob_vol", {})]
        variants += [(f"E) Capped Kelly f={f}", "kelly", {"kelly_fraction": f}) for f in cfg.sizing.kelly_fractions]
        for label, method, kw in variants:
            p = p_kelly if method == "kelly" else ev["proba"]
            # Eşik kapısı tüm yöntemlerde ham olasılıkla aynıdır; yalnızca büyüklük değişir
            gate = ev["proba"] > cfg.meta_model.threshold
            size = position_sizes(method, p, ev, cfg, threshold=-1.0, **kw) * gate
            run = self.run_signal(f"sizing={label}", ev["side"] * size, ev["t1"], register=True)
            rows[label] = {**{k: run.metrics[k] for k in ("Trades", "CAGR", "Sharpe", "Max Drawdown", "Turnover", "Exposure")},
                           "Avg size": float(size[size > 0].mean()) if (size > 0).any() else 0.0,
                           "Max size": float(size.max())}
        self.results["sizing"] = pd.DataFrame(rows).T
        self.results["sizing_note"] = (
            "Mevcut bet sizing (Prado, AFML 10.1): z = (p - 0.5)/sqrt(p(1-p)), m = 2Φ(z) - 1, "
            f"pozisyon = side x m, yalnızca p > {cfg.meta_model.threshold}. Kelly: f = fraction x max(0, p - (1-p)/b), "
            f"b = pt/sl = {cfg.barriers.pt_mult / cfg.barriers.sl_mult:.2f}; Kelly isotonic-kalibre olasılık kullanır "
            f"({n_fallback} olayda kalibrasyon yok -> ham olasılık). Tüm yöntemler max_position="
            f"{cfg.risk.max_position} ile sınırlı; kaldıraç yok."
        )
        return self.results["sizing"]

    # ================================================================ 17
    def backtest(self) -> pd.DataFrame:
        if "thresholds" in self.results:
            return self.results["thresholds"]
        self.benchmarks()
        cfg, ev = self.cfg, self.oos_events
        rows = []
        for thr in cfg.analysis.threshold_grid:
            taken = ev["proba"] > thr
            run = self.run_signal(f"threshold={thr:.2f}", self.meta_signal(ev["proba"], threshold=thr),
                                  ev["t1"], register=True)
            rows.append({"threshold": thr, "Trades": int(taken.sum()), "Coverage": taken.mean(),
                         "Precision": ev["bin"][taken].mean() if taken.any() else np.nan,
                         "Recall": ev["bin"][taken].sum() / max(ev["bin"].sum(), 1),
                         "Win Rate": run.metrics["Win Rate"],
                         "Avg Return": float(run.trades.mean()) if len(run.trades) else np.nan,
                         "Sharpe": run.metrics["Sharpe"], "Max Drawdown": run.metrics["Max Drawdown"],
                         "Turnover": run.metrics["Turnover"]})
        self.results["thresholds"] = pd.DataFrame(rows).set_index("threshold")

        # Long / short ayrımı (meta stratejisi)
        meta_led = self._runs["Meta"].ledger
        total_pnl = meta_led["net"].sum()
        ls = {}
        for label, s in (("Long", 1), ("Short", -1)):
            sub = ev[ev["side"] == s]
            taken = sub["proba"] > cfg.meta_model.threshold
            run = self.run_signal(label, self.meta_signal(sub["proba"], sub), sub["t1"])
            pnl = meta_led.loc[meta_led["side"] == s, "net"].sum()
            ls[label] = {"Trades": run.metrics["Trades"], "Win Rate": run.metrics["Win Rate"],
                         "Avg Return": float(run.trades.mean()) if len(run.trades) else np.nan,
                         "Sharpe": run.metrics["Sharpe"], "Max Drawdown": run.metrics["Max Drawdown"],
                         "Profit Factor": run.metrics["Profit Factor"],
                         "Precision": sub["bin"][taken].mean() if taken.any() else np.nan,
                         "Recall": sub["bin"][taken].sum() / max(sub["bin"].sum(), 1),
                         "AUC": safe_auc(sub["bin"], sub["proba"]),
                         "PnL contribution": pnl / total_pnl if total_pnl != 0 else np.nan}
        self.results["long_short"] = pd.DataFrame(ls).T

        dd, worst = {}, {}
        dist = {}
        for name in MAIN:
            stats, w5 = drawdown_summary(self._runs[name].port.returns)
            dd[name], worst[name] = stats, w5
            dist[name] = trade_distribution(self._runs[name].trades)
        self.results["drawdowns"] = pd.DataFrame(dd).T
        self.results["worst_drawdowns"] = worst
        self.results["trade_distribution"] = pd.DataFrame(dist).T
        return self.results["thresholds"]

    # ================================================================ 18
    def regime_analysis(self) -> dict[str, pd.DataFrame]:
        if "regimes" in self.results:
            return self.results["regimes"]
        cfg, ev = self.cfg, self.oos_events
        thr = cfg.meta_model.threshold
        reg = self.regimes.reindex(ev.index)
        out = {}
        for col in reg.columns:
            rows = {}
            for label in [x for x in reg[col].unique() if x != UNKNOWN] + [UNKNOWN]:
                sub = ev[reg[col] == label]
                if sub.empty:
                    continue
                taken = sub["proba"] > thr
                run = self.run_signal(f"{col}={label}", self.meta_signal(sub["proba"], sub), sub["t1"])
                rows[label] = {"Events": len(sub), "Trades": run.metrics["Trades"], "Win Rate": run.metrics["Win Rate"],
                               "Avg Return": float(run.trades.mean()) if len(run.trades) else np.nan,
                               "Sharpe": run.metrics["Sharpe"], "Max Drawdown": run.metrics["Max Drawdown"],
                               "AUC": safe_auc(sub["bin"], sub["proba"]),
                               "Precision": sub["bin"][taken].mean() if taken.any() else np.nan,
                               "Primary precision": sub["bin"].mean(), "Coverage": taken.mean()}
            out[col] = pd.DataFrame(rows).T
        self.results["regimes"] = out

        # Dönem analizi: ana stratejilerin portföy getirileri dönemlere dilimlenir
        periods = []
        for label, start, end in cfg.analysis.periods:
            row = {"period": label}
            sub = ev.loc[start:end]
            row["Events"] = len(sub)
            row["AUC"] = safe_auc(sub["bin"], sub["proba"]) if len(sub) else np.nan
            taken = sub["proba"] > thr
            row["Precision"] = sub["bin"][taken].mean() if taken.any() else np.nan
            row["Primary precision"] = sub["bin"].mean() if len(sub) else np.nan
            for name in MAIN:
                r = self._runs[name].port.returns.loc[start:end]
                row[f"{name} Sharpe"] = float(r.mean() / r.std() * np.sqrt(self.ppy)) if len(r) > 20 and r.std() > 0 else np.nan
                row[f"{name} Return"] = float((1 + r).prod() - 1) if len(r) else np.nan
            row["Meta trades"] = int((self._runs["Meta"].ledger.index.to_series().between(start, end)).sum())
            periods.append(row)
        self.results["periods"] = pd.DataFrame(periods).set_index("period")
        return out

    # ================================================================ 19
    def feature_importance(self) -> pd.DataFrame:
        if "importance" in self.results:
            return self.results["importance"]
        cfg, ds, ev = self.cfg, self.dataset, self.oos_events
        mdi = mdi_importance(self.oos.models, ds.X.columns)
        perm = permutation_importance_oos(self.oos.models, self.oos.splits, ds.X, ds.y,
                                          cfg.analysis.permutation_repeats, self.rng(3000))
        shap, shap_method = shap_importance_oos(self.oos.models, self.oos.splits, ds.X)
        self.results["importance"] = pd.concat([mdi, perm, shap], axis=1).sort_values("Permutation (ΔAUC)", ascending=False)
        self.results["shap_method"] = shap_method
        self.results["feature_corr"] = ds.X.drop(columns=["side"], errors="ignore").corr(method="spearman")

        rows = {}
        variants = {"All features": []}
        variants.update({f"No {g}": list(cols) for g, cols in cfg.analysis.feature_groups.items()})
        for name, drop in variants.items():
            if name == "All features":
                p = ev["proba"]
            else:
                cols = [c for c in ds.X.columns if c not in drop]
                p = walk_forward(ds, cfg, X=ds.X[cols]).proba.reindex(ev.index)
            taken = p > cfg.meta_model.threshold
            run = self.run_signal(f"ablation={name}", self.meta_signal(p), ev["t1"], register=True)
            rows[name] = {"Dropped": ", ".join(drop) or "-", "AUC": safe_auc(ev["bin"], p),
                          "Precision": ev["bin"][taken].mean() if taken.any() else np.nan,
                          "Trades": run.metrics["Trades"], "Sharpe": run.metrics["Sharpe"],
                          "Max Drawdown": run.metrics["Max Drawdown"]}
        self.results["ablation"] = pd.DataFrame(rows).T
        return self.results["importance"]

    # ================================================================ 20
    def statistical_tests(self) -> pd.DataFrame:
        if "stats" in self.results:
            return self.results["stats"]
        self.benchmarks()
        a = self.cfg.analysis
        trial_srs = np.array(list(self.trials.values()))
        n_trials = len(trial_srs)
        rows = {}
        for name in MAIN:
            run = self._runs[name]
            r = run.port.returns
            block = bootstrap_returns(r, a.bootstrap_n, self.rng(4000), a.bootstrap_block, self.ppy)
            iid = bootstrap_returns(r, a.bootstrap_n, self.rng(4001), None, self.ppy)
            years = len(r) / self.ppy
            tb = bootstrap_trades(run.trades, a.bootstrap_n, self.rng(4002), len(run.trades) / years)
            rows[name] = {
                "Sharpe": run.metrics["Sharpe"],
                "Block CI low": block["sharpe_ci_low"], "Block CI high": block["sharpe_ci_high"],
                "IID CI low": iid["sharpe_ci_low"], "IID CI high": iid["sharpe_ci_high"],
                "Trade CI low": tb.get("sharpe_ci_low", np.nan), "Trade CI high": tb.get("sharpe_ci_high", np.nan),
                "P(Sharpe>0) block": block["p_sharpe_gt_0"],
                "P(CAGR<0) block": block["p_cagr_lt_0"],
                "PSR": run.metrics["PSR"],
                "DSR": deflated_sharpe(r, trial_srs, n_trials),
                "N trials": n_trials,
            }
            run.metrics["DSR"] = rows[name]["DSR"]
        self.results["stats"] = pd.DataFrame(rows).T
        bench = self.results["benchmarks"]
        for label, run in self._bench_runs.items():  # E/F dağılım medyanlarıdır; DSR tanımsız
            bench.loc[label, "DSR"] = deflated_sharpe(run.port.returns, trial_srs, n_trials)
        self.results["bootstrap_note"] = (
            f"Block bootstrap: durağan bootstrap, ort. blok {a.bootstrap_block} bar, {a.bootstrap_n} tekrar. "
            "IID bootstrap otokorelasyonu ve volatilite kümelenmesini yok sayar, güven aralığını olduğundan "
            "dar gösterebilir; trade bootstrap örtüşen işlemleri bağımsız varsayar. Karar için block sonucu esas alınır. "
            f"DSR, bu çalıştırmada OOS'ta değerlendirilen {n_trials} strateji varyantına göre düzeltilmiştir; "
            "önceki deneyler (experiment log) dahil edilirse DSR daha da düşer."
        )
        return self.results["stats"]

    # ================================================================ 21
    def cpcv(self) -> pd.DataFrame:
        if "cpcv" in self.results:
            return self.results["cpcv"]
        cfg, ds = self.cfg, self.dataset
        cp = CombinatorialPurgedKFold(ds.label_end, cfg.cpcv.n_groups, cfg.cpcv.n_test_groups, cfg.cv.embargo_pct)
        est = make_estimator(ds, cfg)
        w = sample_weight(ds, cfg.meta_model.weighting).to_numpy()
        combos, preds, splits = [], [], []
        for combo, tr, te in cp.split():
            combos.append(combo)
            splits.append((tr, te))
            model = clone(est).fit(ds.X.iloc[tr], ds.y.iloc[tr], sample_weight=w[tr])
            preds.append(pd.Series(positive_class_proba(model, ds.X.iloc[te]), index=ds.X.index[te]))
        self.results["cpcv_splits"] = splits
        ev = ds.events
        window = (ev.index.min(), ev["exit_time"].max())
        rows = []
        for p_i, path in enumerate(cp.paths(combos)):
            proba = pd.concat([preds[s].loc[ds.X.index[cp.groups[g]]] for g, s in sorted(path.items())])
            for name, method in (("Meta", "equal"), ("Meta+Sizing", cfg.sizing.method)):
                run = self.run_signal(f"cpcv{p_i}", self.meta_signal(proba, ev, method=method), ev["t1"], window=window)
                rows.append({"path": p_i, "strategy": name, "Sharpe": run.metrics["Sharpe"],
                             "CAGR": run.metrics["CAGR"], "Max Drawdown": run.metrics["Max Drawdown"],
                             "AUC": safe_auc(ev["bin"], proba)})
        prim = self.run_signal("cpcv_primary", ev["side"].astype(float), ev["t1"], window=window)
        self.results["cpcv"] = pd.DataFrame(rows)
        self.results["cpcv_info"] = {"n_groups": cfg.cpcv.n_groups, "k": cfg.cpcv.n_test_groups,
                                     "n_splits": len(combos), "n_paths": cp.n_paths,
                                     "primary_sharpe_same_window": prim.metrics["Sharpe"]}
        return self.results["cpcv"]

    # ================================================================ 22
    def randomization_tests(self) -> pd.DataFrame:
        if "placebo" in self.results:
            return self.results["placebo"]
        cfg, ds, ev = self.cfg, self.dataset, self.oos_events
        a = cfg.analysis
        real = self._runs["Meta"].metrics["Sharpe"]
        dists: dict[str, list[float]] = {}

        dists["A) Randomized labels"] = [m["Sharpe"] for m in self.randomized_label_runs()]

        # D) Öznitelik kolonları bağımsız karıştırılır (side korunur)
        d = []
        feat_cols = [c for c in ds.X.columns if c != "side"]
        for r in range(a.placebo_feature_reps):
            rng = self.rng(5000 + r)
            Xp = ds.X.copy()
            for c in feat_cols:
                Xp[c] = rng.permutation(Xp[c].to_numpy())
            p = walk_forward(ds, cfg, X=Xp, seed=cfg.experiment.seed + r).proba.reindex(ev.index)
            d.append(self.run_signal("placebo_feat", self.meta_signal(p), ev["t1"]).metrics["Sharpe"])
        dists["D) Randomized feature columns"] = d

        # C) Rastgele olay zamanları (aynı sayıda, birincil yönü olan barlardan)
        c_list = []
        index = self.ohlcv.index
        valid = self.market.features.notna().all(axis=1) & (self.market.side != 0) & (
            self.market.vol > cfg.barriers.min_target)
        valid.iloc[-(cfg.barriers.max_holding_bars + 2):] = False
        pool = index[valid.to_numpy()]
        for r in range(a.placebo_timestamp_reps):
            rng = self.rng(6000 + r)
            t_ev = pool[np.sort(rng.choice(len(pool), size=min(len(self.t_events), len(pool)), replace=False))]
            try:
                ds_r = build_event_dataset(self.market, cfg, t_ev, self.member_mask)
                oos = walk_forward(ds_r, cfg, seed=cfg.experiment.seed + r)
                ev_r = ds_r.events.loc[oos.proba.notna()]
                win = (ev_r.index.min(), ev_r["exit_time"].max())
                sig = self.meta_signal(oos.proba.reindex(ev_r.index), ev_r)
                c_list.append(self.run_signal("placebo_ts", sig, ev_r["t1"], window=win).metrics["Sharpe"])
            except ValueError as exc:
                self.log.warning("Placebo C tekrar %d atlandı: %s", r, exc)
        dists["C) Randomized event timestamps"] = c_list

        # B) Karıştırılmış getiriler: bar göreli OHLCV satırları karıştırılıp fiyat yolu yeniden kurulur
        b_list = []
        for r in range(a.placebo_return_reps):
            synth = shuffle_bars(self.ohlcv, self.rng(7000 + r))
            try:
                m = build_market_state(synth, cfg, adv_value(synth, cfg.costs.adv_window))
                ds_r = build_event_dataset(m, cfg)
                oos = walk_forward(ds_r, cfg, seed=cfg.experiment.seed + r)
                ev_r = ds_r.events.loc[oos.proba.notna()]
                win = (ev_r.index.min(), ev_r["exit_time"].max())
                sig = self.meta_signal(oos.proba.reindex(ev_r.index), ev_r)
                b_list.append(self.run_signal("placebo_ret", sig, ev_r["t1"], window=win, ohlcv=synth,
                                              adv=m.adv).metrics["Sharpe"])
            except ValueError as exc:
                self.log.warning("Placebo B tekrar %d atlandı: %s", r, exc)
        dists["B) Shuffled returns"] = b_list

        rows = {}
        for name, vals in sorted(dists.items()):
            v = np.array([x for x in vals if np.isfinite(x)])
            p = (1 + np.sum(v >= real)) / (1 + len(v)) if len(v) else np.nan
            rows[name] = {"Real Sharpe": real, "Placebo mean": v.mean() if len(v) else np.nan,
                          "Placebo 95%": np.percentile(v, 95) if len(v) else np.nan,
                          "Reps": len(v), "p-value": p}
        self.results["placebo"] = pd.DataFrame(rows).T
        self.results["placebo_dists"] = dists
        return self.results["placebo"]

    # ================================================================ 23
    def visualizations(self) -> dict[str, Path]:
        from . import plots

        if not self.cfg.experiment.save_figures:
            return {}
        fig_dir = self.out_dir / "figures"
        figs = plots.make_all(self, fig_dir)
        self.results["figures"] = figs
        return figs

    # ================================================================ 24
    def final_report(self):
        from .report import build_report

        report = build_report(self)
        self.results["report"] = report
        return report

    # ================================================================ tümü
    def run_all(self):
        steps = [
            self.validate_data, self.load_data, self.build_features, self.leakage_audit, self.sample_events,
            self.triple_barrier, self.primary_model, self.meta_labels, self.cross_validation, self.walk_forward,
            self.calibration, self.benchmarks, self.transaction_costs, self.position_sizing, self.backtest,
            self.regime_analysis, self.feature_importance, self.statistical_tests, self.cpcv,
            self.randomization_tests,
        ]
        for step in steps:
            self.log.info("▶ %s", step.__name__)
            step()
        self.leakage_audit(full=True)
        self.visualizations()
        return self.final_report()

    def core_summary(self) -> dict[str, Any]:
        """Çoklu varlık taraması için hafif özet (placebo/CPCV olmadan)."""
        for step in (self.validate_data, self.build_features, self.leakage_audit, self.sample_events,
                     self.triple_barrier, self.walk_forward):
            step()
        ev = self.oos_events
        thr = self.cfg.meta_model.threshold
        prim = self.run_signal("Primary", ev["side"].astype(float), ev["t1"])
        meta = self.run_signal("Meta", self.meta_signal(ev["proba"]), ev["t1"])
        siz = self.run_signal("Meta+Sizing", self.meta_signal(ev["proba"], method=self.cfg.sizing.method), ev["t1"])
        bh = self.run_target("Buy&Hold", pd.Series(1.0, index=self.ohlcv.index))
        taken = ev["proba"] > thr
        return {
            "ticker": self.ticker, "source": self.market_data.source, "bars": len(self.ohlcv),
            "events": len(self.dataset.events), "oos_events": len(ev), "Y=1 rate": ev["bin"].mean(),
            "AUC": safe_auc(ev["bin"], ev["proba"]),
            "Primary precision": ev["bin"].mean(),
            "Meta precision": ev["bin"][taken].mean() if taken.any() else np.nan,
            "Primary Sharpe": prim.metrics["Sharpe"], "Meta Sharpe": meta.metrics["Sharpe"],
            "Meta+Sizing Sharpe": siz.metrics["Sharpe"], "Buy&Hold Sharpe": bh.metrics["Sharpe"],
            "Meta PSR": meta.metrics["PSR"], "Meta trades": meta.metrics["Trades"],
            "Leakage": self.leakage["features"].status,
        }


def _breakeven(sharpe_by_cost: pd.Series) -> float:
    """Meta Sharpe'ın sıfırın altına düştüğü maliyet (doğrusal enterpolasyon)."""
    s = sharpe_by_cost.dropna().sort_index()
    if s.empty:
        return float("nan")
    if (s > 0).all():
        return float("inf")
    if s.iloc[0] <= 0:
        return 0.0
    i = int(np.argmax(s.to_numpy() <= 0))
    x0, x1, y0, y1 = s.index[i - 1], s.index[i], s.iloc[i - 1], s.iloc[i]
    return float(x0 + (x1 - x0) * y0 / (y0 - y1))


def shuffle_bars(ohlcv: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Placebo B: bar göreli OHLC ve hacim satırlarını karıştırıp fiyat yolunu yeniden kurar.

    Getirilerin marjinal dağılımı korunur, zamansal yapı (trend, rejim, oynaklık
    kümelenmesi) yok edilir. Bu yapıya dayanan bir edge burada kaybolmalıdır.
    """
    c = ohlcv["Close"].to_numpy()
    prev = np.concatenate([[c[0]], c[:-1]])
    rel = np.column_stack([ohlcv["Open"] / prev, ohlcv["High"] / prev, ohlcv["Low"] / prev, c / prev])[1:]
    vol = ohlcv["Volume"].to_numpy()[1:]
    perm = rng.permutation(len(rel))
    rel, vol = rel[perm], vol[perm]
    closes = c[0] * np.cumprod(rel[:, 3])
    prevs = np.concatenate([[c[0]], closes[:-1]])
    out = pd.DataFrame(
        {"Open": rel[:, 0] * prevs, "High": rel[:, 1] * prevs, "Low": rel[:, 2] * prevs,
         "Close": closes, "Volume": vol},
        index=ohlcv.index[1:],
    )
    first = ohlcv.iloc[:1]
    return pd.concat([first, out])


def run_universe(cfg: ResearchConfig, tickers: list[str] | None = None) -> pd.DataFrame:
    """Birden çok hisse üzerinde çekirdek OOS testini çalıştırır (her hisse bağımsız)."""
    tickers = list(tickers or cfg.data.tickers)
    rows = []
    for t in tickers:
        try:
            rows.append(ResearchSession(cfg, t).core_summary())
        except (FileNotFoundError, ValueError) as exc:
            logging.getLogger("meta_labeling.research").warning("%s atlandı: %s", t, exc)
            rows.append({"ticker": t, "error": str(exc)})
    return pd.DataFrame(rows).set_index("ticker")
