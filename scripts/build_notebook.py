"""``meta_labeling`` paketinden kendi kendine yeten Jupyter çıktıları üretir.

Üretilen dosyalar (repo klonlanmadan da çalışır, paket import'u gerektirmez):

* ``notebooks/meta_labeling_pipeline.ipynb`` : adım adım hücreler + grafik
* ``notebooks/meta_labeling_tek_hucre.py``   : tek hücreye yapıştırılabilir sürüm

Notebook'taki kod, paket modüllerinden birebir kopyalanır (göreli import'lar ve
modül docstring'leri dışında). Paket değiştiğinde yeniden üretin:

    python scripts/build_notebook.py

``tests/test_notebook.py`` üretilen dosyaların güncel olduğunu doğrular.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "meta_labeling"
OUT_DIR = ROOT / "notebooks"
NOTEBOOK_PATH = OUT_DIR / "meta_labeling_pipeline.ipynb"
SCRIPT_PATH = OUT_DIR / "meta_labeling_tek_hucre.py"

# Bağımlılık sırasına göre modüller ve bölüm başlıkları
MODULES: list[tuple[str, str]] = [
    ("config", "Konfigürasyon"),
    ("data", "Veri: sentetik OHLCV simülasyonu ve CSV yükleyici"),
    ("volatility", "Dinamik volatilite"),
    ("primary", "Birincil model (yön sinyali)"),
    ("sampling", "Olay örnekleme (CUSUM filtresi)"),
    ("labeling", "Triple Barrier Method ve meta-etiketler"),
    ("sample_weights", "Örtüşen etiketler için örnek ağırlıkları"),
    ("features", "Meta-model öznitelikleri (rejim göstergeleri)"),
    ("model", "Meta-model (LightGBM / Random Forest)"),
    ("cv", "Purged & Embargoed çapraz doğrulama"),
    ("sizing", "Sinyal filtreleme ve bet sizing"),
    ("backtest", "Backtest ve performans metrikleri"),
    ("pipeline", "Pipeline orkestrasyonu ve özet rapor"),
]

INTRO_MD = """\
# Meta-Labeling + Triple Barrier ML Pipeline

Marcos López de Prado, *Advances in Financial Machine Learning* metodolojisiyle
aşırı öğrenmeye dirençli, uçtan uca bir işlem pipeline'ı.

**Kullanım:** Hücreleri yukarıdan aşağıya sırayla çalıştırın (*Run → Run All Cells*).
Tanım hücreleri yalnızca fonksiyon/sınıf tanımlar; sonuçlar en alttaki **Çalıştır**
bölümünde üretilir.

**Gereksinimler:** `numpy`, `pandas`, `scikit-learn`, `lightgbm` (grafik için opsiyonel
`matplotlib`). macOS'ta LightGBM import hatası alırsanız `brew install libomp` kurun ya
da Çalıştır bölümünde `kind="rf"` seçin.

> Bu notebook `scripts/build_notebook.py` tarafından `meta_labeling/` paketinden
> otomatik üretilmiştir; kod paketle birebir aynıdır."""

SETUP_CODE = """\
# Gerekirse bir kez çalıştırın (paketler kuruluysa atlayabilirsiniz):
# %pip install -q numpy pandas scikit-learn lightgbm matplotlib

from __future__ import annotations  # tip ipuçları için; IPython sonraki hücrelere de uygular

import warnings

warnings.filterwarnings("ignore", category=UserWarning)"""

RUN_MD = """\
## Çalıştır

Varsayılan ayarlar sentetik veri üretir. Parametreleri `replace(...)` ile değiştirebilirsiniz.
Eşik ve hiperparametreleri OOS sonuçlarına bakarak tekrar tekrar ayarlamak *backtest
overfitting*'e yol açar; değişiklikleri ex-ante yapın."""

RUN_CODE = """\
from dataclasses import replace

cfg = PipelineConfig()
# Örnek değişiklikler:
# cfg = replace(cfg, primary=replace(cfg.primary, kind="bollinger"))
# cfg = replace(cfg, model=replace(cfg.model, kind="rf", threshold=0.60))
# cfg = replace(cfg, barrier=replace(cfg.barrier, pt_mult=1.0, sl_mult=2.0, max_holding_bars=5))

pipeline = MetaLabelingPipeline(cfg)
result = pipeline.run()  # gerçek veri için: pipeline.run(ohlcv)
print(result.summary())"""

TABLES_CODE = """\
# Detay tabloları
display(result.comparison)
display(result.cv_scores)
display(result.events.tail(10))"""

PLOT_MD = """\
### Equity eğrileri (OOS dönem)

Aynı OOS olay kümesi üzerinde üç stratejinin 1 birimlik sermayesinin gelişimi
(günlük portföy getirileri, maliyet dahil)."""

PLOT_CODE = """\
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Grafik için: %pip install matplotlib")

if plt is not None:
    SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
    COLORS = {"primary": "#2a78d6", "filtered": "#eb6834", "sized": "#1baf7a"}

    close = result.ohlcv["Close"]
    oos = result.events.dropna(subset=["proba"])
    m = result.config.model
    signals = meta_signals(oos["side"], oos["proba"], m.threshold, m.step_size)
    start, end = oos.index.min(), oos["t1"].max()

    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []
    for name, sig in signals.items():
        r = portfolio_returns(close, oos["t1"], sig, m.cost_per_side).loc[start:end]
        equity = (1.0 + r).cumprod()
        ax.plot(equity.index, equity.values, color=COLORS[name], lw=2,
                solid_joinstyle="round", solid_capstyle="round", label=STRATEGY_LABELS[name])
        ax.plot(equity.index[-1], equity.iloc[-1], "o", ms=8, color=COLORS[name],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ends.append((equity.iloc[-1], equity.index[-1]))

    # Uç etiketleri: değer metni mürekkep renginde, çakışmaları önlemek için aralıklandırılır
    fig.canvas.draw()
    to_pt = lambda v: ax.transData.transform((0, v))[1] * 72.0 / fig.dpi  # veri -> punto
    last_pt = -1e9
    for value, date in sorted(ends, key=lambda e: e[0]):
        y_pt = max(to_pt(value), last_pt + 13)
        last_pt = y_pt
        ax.annotate(f"x{value:.2f}", xy=(date, value), xytext=(10, y_pt - to_pt(value)),
                    textcoords="offset points", va="center", fontsize=10, color=INK)

    ax.axhline(1.0, color=INK_MUTED, lw=1, alpha=0.6)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    ax.set_ylabel("Sermaye (başlangıç = 1)", color=INK_MUTED)
    ax.set_title("OOS equity eğrileri: birincil model vs meta-labeling", color=INK, loc="left")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    ax.margins(x=0.08)
    plt.tight_layout()
    plt.show()"""

LIVE_MD = """\
### Canlı karar

Tüm etiketli veriyle eğitilmiş `final_model`, etiketi henüz oluşmamış en güncel olayları
puanlar. `signal = side x size`; `size = 0` ise meta-model işlemi onaylamıyor demektir."""

LIVE_CODE = """\
pipeline.score_events(result.ohlcv, result.final_model, last_n=5)"""

REAL_DATA_MD = """\
### Gerçek BIST verisiyle çalıştırma

Aşağıdaki hücrelerden birinin yorumunu kaldırın. CSV için kolonlar:
`Date, Open, High, Low, Close, Volume` (büyük/küçük harf duyarsız)."""

REAL_DATA_CODE = """\
# --- Seçenek 1: CSV dosyası
# ohlcv = load_ohlcv_csv("SASA.csv")

# --- Seçenek 2: yfinance (%pip install yfinance)
# import yfinance as yf
# raw = yf.download("SASA.IS", period="10y", interval="1d", auto_adjust=True, progress=False)
# if isinstance(raw.columns, pd.MultiIndex):
#     raw.columns = raw.columns.get_level_values(0)
# ohlcv = raw[["Open", "High", "Low", "Close", "Volume"]]

# real = MetaLabelingPipeline(cfg).run(ohlcv)
# print(real.summary())"""

SCRIPT_HEADER = """\
# ============================================================
# META-LABELING + TRIPLE BARRIER ML PIPELINE (AFML / López de Prado)
# Tek hücre: Kopyala -> Jupyter hücresine yapıştır -> Çalıştır
# Gereksinimler: %pip install numpy pandas scikit-learn lightgbm
#
# Bu dosya scripts/build_notebook.py tarafından meta_labeling/ paketinden
# otomatik üretilmiştir; elle düzenlemeyin.
# ============================================================"""

SCRIPT_RUN = """\
# ============================================================
# ÇALIŞTIR
# ============================================================
from dataclasses import replace

cfg = PipelineConfig()
# cfg = replace(cfg, primary=replace(cfg.primary, kind="bollinger"))
# cfg = replace(cfg, model=replace(cfg.model, kind="rf", threshold=0.60))

pipeline = MetaLabelingPipeline(cfg)
result = pipeline.run()  # gerçek veri: pipeline.run(load_ohlcv_csv("SASA.csv"))
print(result.summary())
print()
print("Canlı karar (son 5 olay):")
print(pipeline.score_events(result.ohlcv, result.final_model, last_n=5))"""


def module_parts(name: str) -> tuple[str, str]:
    """(modül docstring'i, göreli import'ları ve docstring'i çıkarılmış kod) döndürür."""
    src = (PKG / f"{name}.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    drop: set[int] = set()
    for i, node in enumerate(tree.body):
        is_docstring = (
            i == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        is_local_import = isinstance(node, ast.ImportFrom) and (
            node.level > 0 or node.module == "__future__"
        )
        if is_docstring or is_local_import:
            drop.update(range(node.lineno, node.end_lineno + 1))
    lines = [line for no, line in enumerate(src.splitlines(), 1) if no not in drop]
    code = "\n".join(lines).strip("\n")
    while "\n\n\n\n" in code:
        code = code.replace("\n\n\n\n", "\n\n\n")
    return doc, code


def _comment(text: str) -> str:
    return "\n".join(f"# {line}".rstrip() for line in text.splitlines())


def _cell(kind: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    cell = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build_notebook() -> str:
    cells = [_cell("markdown", INTRO_MD), _cell("code", SETUP_CODE)]
    for step, (name, title) in enumerate(MODULES, 1):
        doc, code = module_parts(name)
        summary = doc.splitlines()[0] if doc else ""
        cells.append(_cell("markdown", f"## {step}. {title}\n\n{summary}"))
        cells.append(_cell("code", f"{_comment(doc)}\n\n{code}" if doc else code))
    cells += [
        _cell("markdown", RUN_MD),
        _cell("code", RUN_CODE),
        _cell("code", TABLES_CODE),
        _cell("markdown", PLOT_MD),
        _cell("code", PLOT_CODE),
        _cell("markdown", LIVE_MD),
        _cell("code", LIVE_CODE),
        _cell("markdown", REAL_DATA_MD),
        _cell("code", REAL_DATA_CODE),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def build_script() -> str:
    parts = [SCRIPT_HEADER, "from __future__ import annotations\n\nimport warnings\n\n"
             'warnings.filterwarnings("ignore", category=UserWarning)']
    for step, (name, title) in enumerate(MODULES, 1):
        doc, code = module_parts(name)
        banner = f"# {'=' * 60}\n# {step}. {title.upper()}\n# {'=' * 60}"
        parts.append(f"{banner}\n{_comment(doc)}\n\n{code}" if doc else f"{banner}\n{code}")
    parts.append(SCRIPT_RUN)
    return "\n\n\n".join(parts) + "\n"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    NOTEBOOK_PATH.write_text(build_notebook(), encoding="utf-8")
    SCRIPT_PATH.write_text(build_script(), encoding="utf-8")
    print(f"Yazıldı: {NOTEBOOK_PATH.relative_to(ROOT)}, {SCRIPT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
