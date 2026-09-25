"""Otomatik araştırma raporu (Markdown) ve nihai tablolar.

Rapor bir yatırım tavsiyesi ya da gelecek getiri tahmini DEĞİLDİR. Amaç,
backtest sonucunun güvenilir olup olmadığını test etmektir: edge'in hangi
koşullarda gözlendiği, hangi testlerde kaybolduğu, hangi risklerin bulunduğu
ve hangi sonuçların istatistiksel olarak belirsiz olduğu raporlanır.

Geçme kriterleri ``config.yaml -> criteria`` bölümünde ex-ante tanımlıdır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .tracking import ExperimentTracker, build_record

if TYPE_CHECKING:  # pragma: no cover
    from .session import ResearchSession

PASS, FAIL, INFO, INCONCLUSIVE = "PASS", "FAIL", "INFO", "INCONCLUSIVE"
VERDICT_FAILED = "Robustness tests failed"
VERDICT_SURVIVED = "Edge survived the robustness battery (istatistiksel kanıt; garanti değil)"
VERDICT_MIXED = "Mixed / statistically inconclusive"
VERDICT_INVALID = "INVALID: leakage detected"

PCT_ROWS = {"CAGR", "Total Return", "Ann. Volatility", "Max Drawdown", "Win Rate", "Exposure", "PSR", "DSR",
            "Coverage", "Precision", "Recall", "Net Return", "P(CAGR<0)", "Primary precision", "Y=1 rate",
            "Meta precision", "Avg Return", "PnL contribution", "P(Sharpe>0) block", "P(CAGR<0) block"}


INT_COLS = {"Trades", "Events", "Reps", "n", "n_train", "n_test", "Episodes", "N trials", "Meta trades", "path",
            "events", "capital", "duration_bars", "recovery_bars"}


@dataclass
class ResearchReport:
    verdict: str
    final_table: pd.DataFrame
    robustness_table: pd.DataFrame
    evidence: dict[str, list[str]]
    markdown: str
    path: Path | None = None
    experiment_path: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ biçimleme
def _fmt(value: Any, pct: bool = False, integer: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if integer and isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return f"{int(value):,}"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if np.isinf(value):
            return "∞" if value > 0 else "-∞"
        return f"{100 * value:.1f}%" if pct else f"{value:.3f}"
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    return str(value)


def md_table(df: pd.DataFrame, pct_cols: set[str] | None = None, index: bool = True) -> str:
    if df is None or df.empty:
        return "_(veri yok)_\n"
    pct_cols = PCT_ROWS if pct_cols is None else pct_cols
    cols = list(df.columns)
    header = ([str(df.index.name or "")] if index else []) + [str(c) for c in cols]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in df.iterrows():
        cells = [_fmt(idx)] if index else []
        cells += [_fmt(row[c], str(c) in pct_cols, str(c) in INT_COLS) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def md_rows_table(df: pd.DataFrame) -> str:
    """Satırları metrik olan tablolar (satır adına göre yüzde biçimi)."""
    if df is None or df.empty:
        return "_(veri yok)_\n"
    header = ["Metric"] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for idx, row in df.iterrows():
        pct, integer = str(idx) in PCT_ROWS, str(idx) in INT_COLS
        lines.append("| " + " | ".join([str(idx)] + [_fmt(v, pct, integer) for v in row]) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ tablolar
def final_metric_table(s: ResearchSession) -> pd.DataFrame:
    stats = s.results["stats"]
    rows = ["CAGR", "Total Return", "Ann. Volatility", "Sharpe", "Sortino", "Max Drawdown", "Calmar",
            "Win Rate", "Profit Factor", "Trades", "Turnover", "Exposure", "PSR", "DSR"]
    data = {}
    for name in ("Primary", "Meta", "Meta+Sizing", "Buy&Hold"):
        m = s._runs[name].metrics
        col = {r: m[r] for r in rows}
        st = stats.loc[name]
        col["DSR"] = st["DSR"]
        col["Sharpe 95% CI (block bootstrap)"] = f"[{st['Block CI low']:.2f}, {st['Block CI high']:.2f}]"
        col["P(CAGR<0)"] = st["P(CAGR<0) block"]
        data[name] = col
    return pd.DataFrame(data)


def _row(test: str, result: str, status: str, interp: str) -> dict[str, str]:
    return {"Robustness Test": test, "Result": result, "Pass/Fail": status, "Interpretation": interp}


def robustness_table(s: ResearchSession) -> tuple[pd.DataFrame, list[str], str]:
    c = s.cfg.criteria
    rows, core = [], []
    leak = s.leakage.get("full") or s.leakage.get("features")
    rows.append(_row("Data leakage audit", f"{len(leak.findings)} kontrol, {len(leak.failures)} FAIL",
                     leak.status, "Özniteliklerde, etiketlerde, CV bölmelerinde ve kalibrasyonda gelecek bilgisi "
                     "tespit edilmedi" if leak.status == PASS else "Sızıntı var: diğer tüm sonuçlar geçersiz"))

    m, p = s._runs["Meta"].metrics, s._runs["Primary"].metrics
    ok = m["Sharpe"] > p["Sharpe"]
    rows.append(_row("Meta vs Primary (OOS Sharpe)", f"{m['Sharpe']:.2f} vs {p['Sharpe']:.2f}", PASS if ok else FAIL,
                     "Meta-filtre birincil sinyale göre risk-ayarlı getiri ekliyor" if ok
                     else "Meta-filtre birincil modele katkı sağlamıyor"))
    core.append("Meta vs Primary (OOS Sharpe)")

    st = s.results["stats"].loc["Meta"]
    ok = st["PSR"] >= c.psr_min
    rows.append(_row("PSR (Meta)", f"{st['PSR']:.3f} (eşik {c.psr_min})", PASS if ok else FAIL,
                     "Sharpe > 0 olasılığı yüksek (tek deneme varsayımıyla)" if ok else "Sharpe'ın 0'dan büyüklüğü belirsiz"))
    ok = st["DSR"] >= c.dsr_min
    rows.append(_row("Deflated Sharpe (Meta)", f"{st['DSR']:.3f} ({int(st['N trials'])} deneme)", PASS if ok else FAIL,
                     "Çoklu deneme düzeltmesinden sonra da anlamlı" if ok
                     else "Çoklu deneme düzeltmesinden sonra anlamlı değil (seçim yanlılığı açıklayabilir)"))
    core.append("Deflated Sharpe (Meta)")
    ok = st["Block CI low"] > 0
    rows.append(_row("Block bootstrap Sharpe CI (Meta)", f"[{st['Block CI low']:.2f}, {st['Block CI high']:.2f}]",
                     PASS if ok else FAIL, "%95 güven aralığı 0'ı içermiyor" if ok
                     else "%95 güven aralığı 0'ı içeriyor: edge istatistiksel olarak belirsiz"))
    core.append("Block bootstrap Sharpe CI (Meta)")

    rand = pd.DataFrame(s.results["random_entry"])["Sharpe"].dropna()
    q = float(np.percentile(rand, 100 * c.random_entry_percentile)) if len(rand) else np.nan
    ok = m["Sharpe"] > q
    rows.append(_row("Random entry benchmark", f"Meta {m['Sharpe']:.2f} vs rastgele %{100 * c.random_entry_percentile:.0f} "
                     f"persentil {q:.2f}", PASS if ok else FAIL,
                     "Rastgele giriş + aynı bariyerlerden ayrışıyor" if ok else "Rastgele girişlerden ayırt edilemiyor"))
    core.append("Random entry benchmark")

    placebo = s.results["placebo"]
    meaning = {
        "A": ("Model, etiket karıştırılmış modellerden ayrışıyor: öğrenilen ilişki şansa bağlı değil",
              "Etiket karıştırılmış (bilgisiz) modellerden ayrışmıyor"),
        "B": ("Edge zamansal yapıya (rejim/trend) dayanıyor; yapı yok edilince kayboluyor",
              "Getiri sırası karıştırılmış piyasalarda da benzer sonuç: edge zamansal yapıya bağlı değil"),
        "C": ("CUSUM olay zamanlaması rastgele zamanlamaya göre katkı sağlıyor",
              "CUSUM zamanlaması rastgele olay zamanlarından üstün değil; edge (varsa) başka bileşenden geliyor"),
        "D": ("Öznitelikler bilgi taşıyor: karıştırılınca performans düşüyor",
              "Öznitelikler karıştırıldığında performans düşmüyor: rejim öznitelikleri bilgi taşımıyor olabilir"),
    }
    for name, row in placebo.iterrows():
        reps = int(row["Reps"])
        min_p = 1.0 / (1 + reps) if reps else 1.0
        good, bad = meaning.get(name[0], ("Gerçek sonuç placebo dağılımından ayrışıyor",
                                          "Gerçek sonuç placebo dağılımından ayrışmıyor"))
        if min_p >= c.placebo_alpha:
            status, interp = INCONCLUSIVE, (f"{reps} tekrar ile ulaşılabilecek en küçük p = {min_p:.3f} >= α; "
                                            "test bu tekrar sayısıyla karar veremez")
        else:
            ok = row["p-value"] < c.placebo_alpha
            status, interp = (PASS, good) if ok else (FAIL, bad)
        rows.append(_row(f"Placebo {name}", f"p = {row['p-value']:.3f} ({reps} tekrar, placebo ort. "
                         f"{row['Placebo mean']:.2f}, gerçek {row['Real Sharpe']:.2f})", status, interp))
        if name.startswith(("A)", "D)")):
            core.append(f"Placebo {name}")

    be = s.results["breakeven_cost_bps"]
    ok = be >= c.min_breakeven_cost_bps
    rows.append(_row("Transaction cost breakeven (Meta)", f"{_fmt(be)} bps/yön (eşik {c.min_breakeven_cost_bps})",
                     PASS if ok else FAIL, "Makul maliyet artışına dayanıklı" if ok
                     else "Edge küçük maliyet artışlarında kayboluyor"))

    cp = s.results["cpcv"]
    v = cp.loc[cp["strategy"] == "Meta", "Sharpe"]
    frac = float((v > 0).mean()) if len(v) else np.nan
    ok = frac >= c.cpcv_min_positive_frac
    rows.append(_row("CPCV path Sharpe > 0", f"{frac:.0%} yol pozitif (medyan {v.median():.2f})", PASS if ok else FAIL,
                     "Farklı backtest yollarında tutarlı" if ok else "Sonuç tek bir tarihsel yola bağımlı olabilir"))

    per = s.results["periods"]
    active = per[per["Meta trades"] >= 5]
    pos = float((active["Meta Return"] > 0).mean()) if len(active) else np.nan
    ok = pos >= c.period_min_positive_frac
    rows.append(_row("Time-period consistency", f"{pos:.0%} dönem pozitif ({len(active)} aktif dönem)",
                     PASS if ok else FAIL, "Edge dönemler arasında yayılmış" if ok
                     else "Edge birkaç döneme yoğunlaşmış olabilir"))

    cal = s.results["calibration"]["metrics"]
    rows.append(_row("Calibration (raw ECE → isotonic ECE)",
                     f"{cal.loc['raw', 'ECE']:.3f} → {cal.loc['isotonic', 'ECE']:.3f}" if "isotonic" in cal.index
                     else f"{cal.loc['raw', 'ECE']:.3f}", INFO,
                     "Ham olasılıklar class_weight=balanced nedeniyle dengelenmiş ölçektedir; mutlak olasılık olarak "
                     "yorumlanmamalı"))
    ls = s.results["long_short"]
    rows.append(_row("Long / Short symmetry", f"Long SR {ls.loc['Long', 'Sharpe']:.2f}, Short SR "
                     f"{ls.loc['Short', 'Sharpe']:.2f}", INFO,
                     "İki taraf arasında belirgin asimetri varsa edge tek yönlü olabilir"))

    table = pd.DataFrame(rows)
    status = dict(zip(table["Robustness Test"], table["Pass/Fail"]))
    core_fail = sum(status.get(t) == FAIL for t in core)
    core_all_pass = all(status.get(t) == PASS for t in core)
    scored = table[table["Pass/Fail"].isin([PASS, FAIL])]
    pass_rate = float((scored["Pass/Fail"] == PASS).mean()) if len(scored) else 0.0
    if leak.status == FAIL:
        verdict = VERDICT_INVALID
    elif core_all_pass and pass_rate >= 0.75:
        verdict = VERDICT_SURVIVED
    elif core_fail >= len(core) / 2 or pass_rate < 0.5:
        verdict = VERDICT_FAILED
    else:
        verdict = VERDICT_MIXED
    return table, core, verdict


def evidence_sections(s: ResearchSession, table: pd.DataFrame, verdict: str) -> dict[str, list[str]]:
    passed = table[table["Pass/Fail"] == PASS]
    failed = table[table["Pass/Fail"] == FAIL]
    strongest = [f"{r['Robustness Test']}: {r['Result']} — {r['Interpretation']}" for _, r in passed.iterrows()]
    weakest = [f"{r['Robustness Test']}: {r['Result']} — {r['Interpretation']}" for _, r in failed.iterrows()]

    risks = []
    if s.market_data.simulated:
        risks.append("Veri SENTETİKTİR: sonuçlar gerçek BIST piyasası hakkında kanıt değildir; simülatörün "
                     "yapısı (rejim değişimi) meta-modelin öğrenebileceği bir sinyal içerecek şekilde tasarlanmıştır.")
    risks.append(s.results["data"]["survivorship"])
    be = s.results["breakeven_cost_bps"]
    if np.isfinite(be) and be < 30:
        risks.append(f"Maliyet hassasiyeti: Meta Sharpe yaklaşık {be:.0f} bps/yön maliyette sıfıra iniyor.")
    n_tr = s._runs["Meta"].metrics["Trades"]
    if n_tr < 300:
        risks.append(f"Düşük örneklem: yalnızca {n_tr} meta işlem; istatistiksel güç sınırlı.")
    per = s.results["periods"]
    if "Meta Return" in per and per["Meta Return"].notna().any():
        top = per["Meta Return"].idxmax()
        share = per["Meta Return"].clip(lower=0).max() / max(per["Meta Return"].clip(lower=0).sum(), 1e-12)
        if share > 0.4:
            risks.append(f"Dönem yoğunlaşması: pozitif getirinin %{100 * share:.0f}'i tek dönemden ({top}).")
    cp = s.results["cpcv"]
    cp_med = cp.loc[cp["strategy"] == "Meta", "Sharpe"].median()
    wf_sr = s._runs["Meta"].metrics["Sharpe"]
    if np.isfinite(cp_med) and wf_sr > 0 and cp_med < 0.75 * wf_sr:
        risks.append(f"Walk-forward Sharpe ({wf_sr:.2f}) CPCV yol medyanının ({cp_med:.2f}) belirgin üzerinde: "
                     "gözlenen tarihsel yol dağılımın iyimser tarafında olabilir.")
    if s.market_data.simulated:
        risks.append("Tek bir simüle fiyat yolu tek bir gerçekleşmedir: aynı üreticiden farklı yollarda (run_universe) "
                     "sonuç belirgin biçimde değişebilir.")
    ls = s.results["long_short"]["PnL contribution"]
    if ls.notna().all() and (ls.abs() > 0.8).any():
        risks.append("Long/short asimetrisi: PnL ağırlıklı olarak tek taraftan geliyor.")
    if s.results["cost_model"]["adv_simulated"]:
        risks.append("Market impact ve ADV simüle hacme dayanıyor (SİMÜLASYON VARSAYIMI).")
    c = s.cfg
    risks.append(f"Parametreler (EMA {c.primary.ema_fast}/{c.primary.ema_slow}, bariyer {c.barriers.pt_mult}σ/"
                 f"{c.barriers.sl_mult}σ, eşik {c.meta_model.threshold}) ex-ante seçildi, ancak literatür/önceki "
                 "denemelerden etkilenmiş olabilir; önceki deneyler DSR'ye dahil edilmedi.")

    unresolved = [
        "Edge gerçek BIST verisinde (survivorship kontrollü, delist dahil evren) korunuyor mu?",
        "Olasılık kalibrasyonu zaman içinde kararlı mı (fold'lar arası ECE değişimi)?",
        "Rejim değişimi sırasında (kriz dönemleri) meta-model davranışı güvenilir mi?",
        "Eşzamanlı işlemlerin ortalanması (avgActiveSignals) gerçek sermaye tahsisini ne kadar temsil ediyor?",
    ]
    if verdict != VERDICT_SURVIVED:
        unresolved.insert(0, "Başarısız testlerin nedeni gerçek bir edge yokluğu mu, yoksa düşük istatistiksel güç mü?")

    nxt = [
        "Aynı config ile gerçek BIST verisi (CSV / yfinance) ve tarihsel endeks üyelik dosyası üzerinde tek seferlik çalıştırma.",
        "Çoklu hisse: run_universe() ile SASA, THYAO, ASELS, BIMAS, AKBNK, GARAN, EREGL, TUPRS üzerinde aynı ex-ante config.",
        "Tüm deney geçmişini (experiments/index.csv) DSR deneme sayısına dahil etme.",
        "Bir sonraki veri döneminde (henüz görülmemiş) gerçek zamanlı paper-trading ile ileriye dönük doğrulama.",
    ]
    if any("Placebo" in w for w in weakest):
        nxt.append("Placebo testlerinde tekrar sayısını artırma (p-değeri çözünürlüğü için).")
    return {"Strongest evidence": strongest or ["Geçen robustness testi yok."],
            "Weakest evidence": weakest or ["Başarısız robustness testi yok."],
            "Major risks": risks, "Unresolved questions": unresolved, "Next experiments": nxt}


# ------------------------------------------------------------------ rapor
def build_report(s: ResearchSession) -> ResearchReport:
    cfg, R = s.cfg, s.results
    final = final_metric_table(s)
    robust, core, verdict = robustness_table(s)
    evidence = evidence_sections(s, robust, verdict)
    d, oos = R["data"], R["oos"]
    fig = {k: Path(v) for k, v in R.get("figures", {}).items()}
    out_dir = s.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def img(key: str, alt: str) -> str:
        return f"\n![{alt}]({fig[key].relative_to(out_dir).as_posix()})\n" if key in fig else ""

    L: list[str] = []
    add = L.append
    add(f"# Research Report — Meta-Labeling + Triple Barrier ({s.ticker})\n")
    add("> Bu rapor bir araştırma/backtest çıktısıdır. Yatırım tavsiyesi değildir ve gelecekteki getiriler hakkında "
        "tahmin içermez. Amaç, backtest sonucunun güvenilir olup olmadığını test etmektir.\n")

    add("## 1. Executive Summary\n")
    add(f"**Sonuç: {verdict}**\n")
    m, p, bh = s._runs["Meta"].metrics, s._runs["Primary"].metrics, s._runs["Buy&Hold"].metrics
    add(f"- OOS dönemi {oos['window'][0].date()} → {oos['window'][1].date()}, {oos['n_oos']} OOS olay, "
        f"meta-model AUC {oos['auc']:.3f}.")
    add(f"- Sharpe: Primary {p['Sharpe']:.2f}, Meta {m['Sharpe']:.2f}, Meta+Sizing "
        f"{s._runs['Meta+Sizing'].metrics['Sharpe']:.2f}, Buy&Hold {bh['Sharpe']:.2f} "
        f"(execution {cfg.execution.mode}, maliyet dahil).")
    n_pass = int((robust["Pass/Fail"] == PASS).sum())
    n_fail = int((robust["Pass/Fail"] == FAIL).sum())
    n_inc = int((robust["Pass/Fail"] == INCONCLUSIVE).sum())
    add(f"- Robustness: {n_pass} PASS, {n_fail} FAIL, {n_inc} INCONCLUSIVE. Çekirdek testler: {', '.join(core)}.")
    add("- Edge'in gözlendiği / kaybolduğu koşullar için bkz. Bölüm 15-21 ve Final bölümündeki kanıt listeleri.\n")

    add("## 2. Dataset\n")
    add(f"- Hisse: **{d['ticker']}**, kaynak: `{d['source']}`, {d['bars']} bar ({d['start']} → {d['end']}).")
    for note in d["notes"]:
        add(f"- ⚠️ {note}")
    add(f"- Survivorship: {d['survivorship']}\n")
    add(md_table(R["validation"].set_index("check"), set()))

    add("## 3. Event Sampling\n")
    e = R["events"]
    add(f"- Mod: `{e['mode']}`, CUSUM eşiği {e['cusum_threshold']} (σ_t: EWMA({cfg.barriers.vol_span}) log-getiri "
        f"volatilitesi, yalnızca t'ye kadar veri). {e['n_events']} olay ({e['long']} long / {e['short']} short), "
        f"100 barda {e['events_per_100_bars']:.1f} olay.\n")

    add("## 4. Triple Barrier Methodology\n")
    b = R["barrier"]
    add(f"- Üst bariyer {cfg.barriers.pt_mult}σ, alt bariyer {cfg.barriers.sl_mult}σ, dikey bariyer "
        f"{cfg.barriers.max_holding_bars} bar, min hedef {cfg.barriers.min_target}.")
    add("- Temas dağılımı: " + ", ".join(f"{k} {v:.1%}" for k, v in b["touch"].items())
        + f"; ortalama elde tutma {b['avg_holding_bars']:.1f} bar; ortalama benzersizlik {b['avg_uniqueness']:.2f}.")
    add(f"- Execution: sinyal close(t), işlem {'open(t+1)' if cfg.execution.mode == 'next_open' else 'close(t)'}; "
        "çıkış bariyer barını izleyen execution fiyatında. Etiket bitişi (purging) = çıkış barı.\n")

    add("## 5. Primary Model\n")
    add(f"- {cfg.primary.kind.upper()} ({cfg.primary.ema_fast}/{cfg.primary.ema_slow}) — yalnızca yön üretir.\n")
    add(md_table(R["primary"]))

    add("## 6. Meta Model\n")
    add(f"- {cfg.meta_model.kind}, sınıf dengeleme (balanced), ağırlıklandırma `{cfg.meta_model.weighting}`, eşik "
        f"p > {cfg.meta_model.threshold} (ex-ante). Hiperparametreler config'den, OOS'a göre ayarlanmadı.")
    add(f"- Meta-etiket: {R['labels']['definition']}. Y=1 oranı {R['labels']['y1_rate']:.1%}.\n")
    add("### Sample weighting (A: none, B: uniqueness)\n")
    add(md_table(R["weighting"]))

    add("## 7. Feature Engineering\n")
    add("Tüm öznitelikler t kapanışına kadar bilinen veriyle hesaplanır (Bölüm 8'de doğrulanmıştır).\n")
    add(md_table(R["features"], set()))

    add("## 8. Leakage Audit\n")
    leak = s.leakage.get("full") or s.leakage["features"]
    add(f"**Durum: {leak.status}** — {len(leak.findings)} kontrol.\n")
    add(md_table(leak.findings[["check", "item", "status", "reason"]].set_index("check"), set()))

    add("## 9. Cross Validation (Purged K-Fold)\n")
    add(f"k={cfg.cv.n_splits}, embargo {cfg.cv.embargo_pct:.0%}. Teşhis amaçlıdır (eğitim test sonrası veriyi de içerir).\n")
    add(md_table(R["cv"][["n_train", "n_test", "auc", "log_loss", "precision", "recall", "base_rate"]], set()))

    add("## 10. Walk Forward OOS\n")
    add(f"- Her fold yalnızca test başlangıcından önce kapanmış etiketlerle eğitildi. OOS olay {oos['n_oos']}, AUC "
        f"{oos['auc']:.3f}, birincil precision {oos['base_precision']:.1%}.\n")

    add("## 11. Benchmark Comparison\n")
    add(R["benchmark_note"] + "\n")
    add(md_table(R["benchmarks"]))

    add("## 12. Calibration\n")
    add(f"Kalibratörler (Platt, Isotonic) her fold için yalnızca geçmiş fold'ların kapanmış OOS etiketleriyle fit "
        f"edildi; {R['calibration']['n_common']} olayda karşılaştırma yapıldı.\n")
    add(md_table(R["calibration"]["metrics"]))
    add(img("calibration", "calibration"))

    add("## 13. Threshold Analysis\n")
    add("Önceden belirlenmiş ızgara; yalnızca raporlama. Eşik bu tabloya bakarak DEĞİŞTİRİLMEZ; her satır DSR'de "
        "deneme olarak sayılır.\n")
    add(md_table(R["thresholds"]))

    add("## 14. Cost Sensitivity\n")
    add(f"Meta breakeven maliyet ≈ **{_fmt(R['breakeven_cost_bps'])} bps/yön**. Model aynı kalır; yalnızca backtest "
        "maliyeti değişir.\n")
    add(md_table(R["cost_sensitivity"].set_index("cost_bps")))
    cm = R["cost_model"]
    add(f"\n**Parametrik maliyet modeli:** `{cm['formula']}`; parametreler: {cm['params']}."
        + (" ⚠️ ADV simüle hacimden hesaplandı (SİMÜLASYON VARSAYIMI)." if cm["adv_simulated"] else "") + "\n")
    add(md_table(cm["table"].set_index("capital")))
    add(img("costs", "cost sensitivity"))

    add("## 15. Regime Analysis\n")
    add("Rejimler yalnızca geçmiş veriyle (genişleyen kantiller) sınıflandırıldı.\n")
    for name, t in R["regimes"].items():
        add(f"### {name}\n")
        add(md_table(t))
    add("### Time periods\n")
    add(md_table(R["periods"]))

    add("## 16. Long/Short Analysis\n")
    add(md_table(R["long_short"]))

    add("## 17. Feature Importance\n")
    add(f"MDI (in-sample, referans), permutation (OOS fold'larda ΔAUC), SHAP ({R['shap_method']}, OOS fold'lar). "
        "Bu sonuçlara göre öznitelik seçimi YAPILMADI.\n")
    add(md_table(R["importance"], set()))
    add("### Ablation (OOS)\n")
    add(md_table(R["ablation"]))
    corr = R["feature_corr"]
    high = [(a, b_, corr.loc[a, b_]) for i, a in enumerate(corr.columns) for b_ in corr.columns[i + 1:]
            if abs(corr.loc[a, b_]) > 0.7]
    add("Yüksek korelasyonlu çiftler (|ρ|>0.7): " + (", ".join(f"{a}/{b_} ({r:.2f})" for a, b_, r in high) or "yok") + "\n")

    add("## 18. Statistical Significance\n")
    add(R["bootstrap_note"] + "\n")
    add(md_table(R["stats"]))

    add("## 19. CPCV\n")
    ci = R["cpcv_info"]
    add(f"N={ci['n_groups']} grup, k={ci['k']} test grubu → {ci['n_splits']} bölme, {ci['n_paths']} backtest yolu. "
        f"Aynı penceredeki Primary Sharpe: {ci['primary_sharpe_same_window']:.2f}.\n")
    cp = R["cpcv"]
    add(md_table(cp.groupby("strategy")[["Sharpe", "CAGR", "Max Drawdown"]].describe()
                 .loc[:, (slice(None), ["mean", "std", "min", "50%", "max"])].round(3), set()))
    add(img("distributions", "CPCV and placebo"))

    add("## 20. Randomization Tests\n")
    add("p-değeri = (1 + #placebo Sharpe ≥ gerçek) / (1 + tekrar). Az tekrar p-değerinin çözünürlüğünü sınırlar.\n")
    add(md_table(R["placebo"], set()))

    add("## 21. Drawdown Analysis\n")
    add(md_table(R["drawdowns"]))
    add("### Worst 5 drawdowns (Meta)\n")
    add(md_table(R["worst_drawdowns"]["Meta"], {"depth"}, index=False))
    add("### Trade distribution\n")
    add(md_table(R["trade_distribution"], {"Mean", "Median", "Std", "Min", "Max", "Avg Win", "Avg Loss"}))
    add(img("trades", "trade distribution"))

    add("## 22. Position Sizing\n")
    add(R["sizing_note"] + "\n")
    add(md_table(R["sizing"]))
    rk = cfg.risk
    add(f"Risk limitleri: max_position={rk.max_position}, max_gross_exposure={rk.max_gross_exposure}, "
        f"max_daily_loss={rk.max_daily_loss}, max_drawdown_stop={rk.max_drawdown_stop}, max_turnover={rk.max_turnover} "
        "(varsayılanlar optimize edilmedi).\n")

    add("## 23. Limitations\n")
    for r in evidence["Major risks"]:
        add(f"- {r}")
    add("- Bariyer temasları kapanış fiyatlarıyla tespit edilir; bar içi (high/low) temaslar modellenmez.")
    add("- Eşzamanlı işlemler ortalanır (avgActiveSignals); gerçek emir büyüklükleri ve kısmi dolum modellenmez.")
    add("- Açığa satış maliyeti/kısıtları (BIST'te açığa satış kuralları, ödünç maliyeti) modellenmez.\n")

    add("## 24. Reproducibility\n")
    add(f"- Config parmak izi `{cfg.fingerprint()}`, tohum {cfg.experiment.seed}. Deney kaydı: "
        f"`{cfg.experiment.output_dir}/experiments/`. `reproduce(<kayıt.json>)` aynı deneyi yeniden çalıştırır.\n")

    add("## 25. Final OOS Results\n")
    add(md_rows_table(final))
    add(img("equity", "equity and drawdown"))
    add("### Robustness summary\n")
    add(md_table(robust, set(), index=False))
    add(f"\n**Karar: {verdict}**\n")
    for title, items in evidence.items():
        add(f"### {title}\n")
        for it in items:
            add(f"- {it}")
        add("")
    markdown = "\n".join(L)

    path = out_dir / "research_report.md"
    path.write_text(markdown, encoding="utf-8")
    final.to_csv(out_dir / "final_oos_results.csv")
    robust.to_csv(out_dir / "robustness_summary.csv", index=False)

    ev = s.dataset.events
    record = build_record(
        cfg=cfg, ticker=s.ticker, ohlcv=s.ohlcv, feature_names=list(s.dataset.X.columns),
        train_period=(ev.index.min(), oos["window"][0]), test_period=oos["window"],
        metrics={"verdict": verdict, "meta_sharpe": m["Sharpe"], "primary_sharpe": p["Sharpe"],
                 "meta_sizing_sharpe": s._runs["Meta+Sizing"].metrics["Sharpe"], "auc": oos["auc"],
                 "final_table": final.to_dict(), "robustness": robust.to_dict(orient="records"),
                 "n_trials": len(s.trials)},
    )
    exp_path = ExperimentTracker(cfg.experiment.output_dir).log(record)
    s.log.info("Rapor: %s | Deney kaydı: %s | Karar: %s", path, exp_path, verdict)
    return ResearchReport(verdict, final, robust, evidence, markdown, path, exp_path)
