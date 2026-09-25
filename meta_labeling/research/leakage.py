"""Otomatik veri sızıntısı (leakage) denetimi.

Denetim türleri
---------------
1. Kesme (truncation) testi: Her öznitelik / volatilite / birincil yön / CUSUM
   olayı / rejim serisi için, örneklenen event zamanlarında t sonrası veri
   silinerek yeniden hesaplanır. t anındaki değer değişiyorsa hesap t sonrası
   bilgi kullanıyordur (ör. ``shift(-1)``, ``center=True`` rolling, tüm
   örneklem normalizasyonu, geleceğe bakan kantil). Bu test rolling, ewm,
   shift, ATR, otokorelasyon, çarpıklık ve hacim özniteliklerinin HEPSİNİ
   aynı kuralla doğrular.
2. Etiket / öznitelik sıralaması: Öznitelik zaman damgası = event zamanı (t0);
   öznitelikler etiket kolonlarından türetilmemiş olmalı (yalnızca OHLCV'den
   yeniden hesaplanınca birebir aynı çıkmalı); execution zamanı sinyalden
   sonra, etiket bitişi execution'dan sonra olmalı.
3. Train/test ayrımı: Her walk-forward bölmesinde tüm eğitim etiketleri test
   başlangıcından ÖNCE kapanmış olmalı; purged k-fold / CPCV bölmelerinde hiçbir
   eğitim etiketi test aralığıyla örtüşmemeli ve embargo uygulanmış olmalı.
4. Kalibrasyon: Kalibratörler yalnızca test fold'undan önce kapanmış
   etiketlerle fit edilmiş olmalı.

Herhangi bir FAIL durumunda ``strict=True`` ise ``LeakageError`` fırlatılır.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("meta_labeling.research")

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"


class LeakageError(RuntimeError):
    """Leakage tespit edildiğinde fırlatılır."""


@dataclass
class LeakageReport:
    findings: pd.DataFrame     # check, item, timestamp, status, reason

    @property
    def status(self) -> str:
        return FAIL if (self.findings["status"] == FAIL).any() else PASS

    @property
    def failures(self) -> pd.DataFrame:
        return self.findings[self.findings["status"] == FAIL]

    def summary(self) -> pd.DataFrame:
        return (
            self.findings.groupby(["check", "status"]).size().unstack(fill_value=0).reset_index()
        )


def _row(check: str, item: str, status: str, reason: str, timestamp=None) -> dict:
    return {"check": check, "item": item, "timestamp": timestamp, "status": status, "reason": reason}


def _as_frame(obj) -> pd.DataFrame:
    return obj.to_frame() if isinstance(obj, pd.Series) else obj


def _show(v) -> str:
    if isinstance(v, str) or pd.isna(v):
        return str(v)
    return f"{float(v):.6g}"


def _same(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return bool(np.isclose(float(a), float(b), rtol=1e-8, atol=1e-12))


def truncation_check(
    check: str,
    fn: Callable[[pd.DataFrame], pd.DataFrame | pd.Series],
    ohlcv: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
) -> list[dict]:
    """fn(ohlcv[:t]).loc[t] == fn(ohlcv).loc[t] olmalı (t sonrası veriden bağımsızlık)."""
    full = _as_frame(fn(ohlcv))
    first_fail: dict[str, dict] = {}
    for t in timestamps:
        part = _as_frame(fn(ohlcv.loc[:t]))
        for col in full.columns:
            if col in first_fail:
                continue
            a = full.at[t, col]
            b = part.at[t, col] if (col in part.columns and t in part.index) else np.nan
            same = _same(a, b)
            if not same:
                first_fail[col] = _row(
                    check, col, FAIL,
                    f"t={t.date()} değeri t sonrası veri silinince değişiyor ({_show(a)} -> {_show(b)}): "
                    "hesap geleceğe ait bilgi kullanıyor",
                    t,
                )
    rows = []
    for col in full.columns:
        rows.append(first_fail.get(col) or _row(check, col, PASS, f"{len(timestamps)} event zamanında kesme testi geçti"))
    return rows


def label_order_checks(events: pd.DataFrame, X: pd.DataFrame, recomputed: pd.DataFrame, mode: str) -> list[dict]:
    rows = []
    check = "label_feature_order"
    if not X.index.equals(events.index):
        rows.append(_row(check, "X.index", FAIL, "Öznitelik zaman damgaları event t0 ile aynı değil"))
    else:
        rows.append(_row(check, "X.index", PASS, "Öznitelikler event zamanında (t0 kapanışı) ölçülüyor"))

    label_cols = {"t1", "bin", "ret", "gross_ret", "net_ret", "exec_net_ret", "exec_gross_ret", "label_end", "barrier"}
    overlap = label_cols & set(X.columns)
    rows.append(
        _row(check, "feature_columns", FAIL if overlap else PASS,
             f"Etiket kolonları öznitelik olarak kullanılmış: {sorted(overlap)}" if overlap
             else "Öznitelik kümesinde etiket kolonu yok")
    )
    cols = [c for c in X.columns if c in recomputed.columns]
    diff = ~np.isclose(X[cols].to_numpy(float), recomputed.reindex(X.index)[cols].to_numpy(float), equal_nan=True)
    if diff.any():
        r, c = np.argwhere(diff)[0]
        rows.append(_row(check, cols[c], FAIL,
                         "Öznitelik yalnızca OHLCV'den yeniden hesaplanınca farklı: etiket/sonrası bilgiye bağımlı",
                         X.index[r]))
    else:
        rows.append(_row(check, "features_vs_labels", PASS,
                         "Öznitelikler etiketlerden bağımsız (yalnızca OHLCV'den yeniden üretilebiliyor)"))

    bad_t1 = events.index[(pd.DatetimeIndex(events["t1"]) <= events.index)]
    rows.append(_row(check, "t1 > t0", FAIL if len(bad_t1) else PASS,
                     "Bariyer teması olay zamanından önce/aynı anda" if len(bad_t1) else "Tüm etiketler t0'dan sonra kapanıyor",
                     bad_t1[0] if len(bad_t1) else None))
    entry = pd.DatetimeIndex(events["entry_time"])
    if mode == "next_open":
        bad = events.index[entry <= events.index]
        rows.append(_row("execution", "entry_time", FAIL if len(bad) else PASS,
                         "İşlem sinyal barında gerçekleşiyor" if len(bad) else "Sinyal close(t), execution open(t+1)",
                         bad[0] if len(bad) else None))
    else:
        rows.append(_row("execution", "entry_time", WARN,
                         "close modu: sinyal kapanışıyla aynı fiyattan işlem varsayılıyor (iyimser, MOC varsayımı)"))
    bad_end = events.index[pd.DatetimeIndex(events["label_end"]) < pd.DatetimeIndex(events["exit_time"])]
    rows.append(_row("execution", "label_end", FAIL if len(bad_end) else PASS,
                     "Etiket bitişi execution çıkışından önce" if len(bad_end) else "Purging etiket bitişini (çıkış barı) kullanıyor",
                     bad_end[0] if len(bad_end) else None))
    return rows


def split_checks(
    name: str,
    label_end: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    walk_forward: bool,
    embargo_pct: float,
) -> list[dict]:
    """Train/test bölmelerinde örtüşme, gelecekten eğitim ve embargo ihlali kontrolü."""
    t0 = label_end.index
    t_end = pd.DatetimeIndex(label_end.to_numpy())
    embargo = (t0[-1] - t0[0]) * embargo_pct
    for k, (train, test) in enumerate(splits):
        if len(train) == 0:
            continue
        test_start = t0[test].min()
        if walk_forward and (t_end[train] >= test_start).any():
            i = train[np.argmax(t_end[train] >= test_start)]
            return [_row(name, f"split {k}", FAIL, "Eğitim etiketi test başlangıcından sonra kapanıyor (geleceğe bakış)", t0[i])]
        # bitişik test blokları (CPCV'de birden fazla blok olabilir)
        blocks = np.split(np.sort(test), np.flatnonzero(np.diff(np.sort(test)) > 1) + 1)
        for blk in blocks:
            s, e = t0[blk[0]], t_end[blk].max()
            overlap = (t0[train] <= e) & (t_end[train] >= s)
            if overlap.any():
                i = train[np.argmax(overlap)]
                return [_row(name, f"split {k}", FAIL, "Eğitim etiketi test etiket aralığıyla örtüşüyor (purging eksik)", t0[i])]
            if not walk_forward:
                emb = (t0[train] > e) & (t0[train] <= e + embargo)
                if emb.any():
                    i = train[np.argmax(emb)]
                    return [_row(name, f"split {k}", FAIL, "Embargo süresi içinde eğitim gözlemi var", t0[i])]
    return [_row(name, "all splits", PASS, f"{len(splits)} bölmede örtüşme/embargo ihlali yok")]


def calibration_checks(fit_windows: dict[str, list[tuple]]) -> list[dict]:
    rows = []
    for method, windows in fit_windows.items():
        bad = [w for w in windows if not w[1] < w[2]]
        rows.append(_row("calibration", method, FAIL if bad else PASS,
                         "Kalibratör test başlangıcından sonra kapanan etiketlerle fit edilmiş" if bad
                         else f"{len(windows)} fold: kalibratör yalnızca geçmiş, kapanmış OOS etiketleriyle fit edildi",
                         bad[0][2] if bad else None))
    return rows


def sample_timestamps(candidates: pd.DatetimeIndex, n: int, rng: np.random.Generator) -> pd.DatetimeIndex:
    if len(candidates) <= n:
        return candidates
    picks = np.sort(rng.choice(len(candidates), size=n - 2, replace=False))
    idx = np.unique(np.concatenate([[0], picks, [len(candidates) - 1]]))
    return candidates[idx]


def finalize(rows: list[dict], strict: bool) -> LeakageReport:
    report = LeakageReport(pd.DataFrame(rows, columns=["check", "item", "timestamp", "status", "reason"]))
    if report.status == FAIL:
        msg = "LEAKAGE TESPİT EDİLDİ:\n" + report.failures.to_string(index=False)
        log.error(msg)
        if strict:
            raise LeakageError(msg)
    else:
        log.info("Leakage audit: PASS (%d kontrol)", len(rows))
    return report


def run_leakage_audit(
    ohlcv: pd.DataFrame,
    cfg,
    dataset=None,
    splits: dict[str, tuple[pd.Series, list, bool]] | None = None,
    calibration_windows: dict[str, list[tuple]] | None = None,
    strict: bool | None = None,
) -> LeakageReport:
    """Tüm leakage kontrollerini çalıştırır ve PASS/FAIL raporu döndürür.

    Args:
        ohlcv: Fiyat verisi.
        cfg: ``ResearchConfig``.
        dataset: ``EventDataset`` (verilirse event zamanları, etiket sırası kontrol edilir).
        splits: {ad: (label_end, [(train, test), ...], walk_forward_mi)}.
        calibration_windows: {yöntem: [(fold, fit_son_etiket, test_başı, n), ...]}.
        strict: True ise FAIL durumunda ``LeakageError`` (varsayılan: config).

    Returns:
        ``LeakageReport``: ``findings`` tablosunda her kontrol için
        check / item (hangi feature) / timestamp / status / reason.
    """
    from ..features import build_feature_matrix
    from ..primary import build_primary_model
    from ..volatility import get_volatility
    from .execution import adv_value
    from .modeling import build_market_state, sample_candidate_events
    from .regimes import classify_regimes

    strict = cfg.experiment.strict_leakage if strict is None else strict
    rng = np.random.default_rng(cfg.experiment.seed)
    b = cfg.barriers

    def vol_fn(d):
        return get_volatility(d["Close"], span=b.vol_span, method=b.vol_method)

    def events_fn(d):
        m = build_market_state(d, cfg, adv_value(d, cfg.costs.adv_window))
        ev = sample_candidate_events(m, cfg)
        return pd.Series(1.0, index=ev).reindex(d.index, fill_value=0.0).rename("cusum_event")

    if dataset is not None:
        candidates = dataset.events.index
    else:
        warm = max(cfg.regimes.min_history, 100)
        candidates = ohlcv.index[warm:]
    ts = sample_timestamps(candidates, cfg.analysis.leakage_samples, rng)

    rows: list[dict] = []
    rows += truncation_check("feature_lookahead", lambda d: build_feature_matrix(d, cfg.features), ohlcv, ts)
    rows += truncation_check("volatility_lookahead", lambda d: vol_fn(d).rename("sigma_ewma"), ohlcv, ts)
    rows += truncation_check("primary_lookahead", lambda d: build_primary_model(cfg.primary_model_config()).side(d), ohlcv, ts)
    rows += truncation_check("event_sampling_lookahead", events_fn, ohlcv, ts)
    rows += truncation_check("adv_lookahead", lambda d: adv_value(d, cfg.costs.adv_window), ohlcv, ts)
    rows += truncation_check("regime_lookahead", lambda d: classify_regimes(d, vol_fn(d), cfg.regimes), ohlcv, ts)

    if dataset is not None:
        recomputed = build_feature_matrix(ohlcv, cfg.features)
        rows += label_order_checks(dataset.events, dataset.X, recomputed, cfg.execution.mode)
    for name, (label_end, sp, wf) in (splits or {}).items():
        rows += split_checks(name, label_end, sp, wf, cfg.cv.embargo_pct)
    if calibration_windows:
        rows += calibration_checks(calibration_windows)
    return finalize(rows, strict)
