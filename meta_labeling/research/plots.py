"""Araştırma grafikleri (matplotlib opsiyonel).

Renkler doğrulanmış kategorik paletten sabit sırayla atanır (strateji kimliği
her grafikte aynı renkte kalır). Her grafikte legend ve/veya doğrudan etiket
bulunur; sayısal değerler rapordaki tablolarda da yer alır.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .session import ResearchSession

SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
STRATEGY_COLORS = {"Primary": "#2a78d6", "Meta": "#eb6834", "Meta+Sizing": "#1baf7a", "Buy&Hold": "#eda100"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
WIN, LOSS = "#2a78d6", "#e34948"


def _plt():
    import matplotlib

    if matplotlib.get_backend().lower() not in ("module://matplotlib_inline.backend_inline", "inline"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax, title: str, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)


def _end_labels(ax, fig, ends: list[tuple[float, pd.Timestamp, str]]) -> None:
    fig.canvas.draw()

    def to_pt(v):
        return ax.transData.transform((0, v))[1] * 72.0 / fig.dpi

    last = -1e9
    for value, date, text in sorted(ends, key=lambda e: e[0]):
        y = max(to_pt(value), last + 12)
        last = y
        ax.annotate(text, xy=(date, value), xytext=(8, y - to_pt(value)), textcoords="offset points",
                    va="center", fontsize=9, color=INK)


def equity_and_drawdown(session: ResearchSession, path: Path) -> Path:
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [2.2, 1]})
    ends = []
    for name, color in STRATEGY_COLORS.items():
        r = session._runs[name].port.returns
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        ax1.plot(eq.index, eq.values, color=color, lw=2, label=name, solid_capstyle="round")
        ax1.plot(eq.index[-1], eq.iloc[-1], "o", ms=7, color=color, mec=SURFACE, mew=2, zorder=3)
        ax2.plot(dd.index, dd.values * 100, color=color, lw=1.5, label=name)
        ends.append((eq.iloc[-1], eq.index[-1], f"{name} x{eq.iloc[-1]:.2f}"))
    ax1.axhline(1.0, color=INK_MUTED, lw=1, alpha=0.5)
    _style(ax1, f"OOS equity eğrileri — {session.ticker} (execution: {session.cfg.execution.mode})",
           "Sermaye (başlangıç = 1)")
    _style(ax2, "Drawdown (underwater)", "%")
    ax1.legend(frameon=False, loc="upper left", labelcolor=INK, fontsize=9)
    ax1.margins(x=0.12)
    _end_labels(ax1, fig, ends)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def calibration_plot(session: ResearchSession, path: Path) -> Path:
    plt = _plt()
    cal = session.results["calibration"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=SURFACE)
    ax1.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls="-", alpha=0.6, label="Mükemmel kalibrasyon")
    for (name, rel), color in zip(cal["reliability"].items(), SERIES):
        ece = cal["metrics"].loc[name, "ECE"]
        ax1.plot(rel["mean_pred"], rel["frac_pos"], "o-", color=color, lw=2, ms=6, mec=SURFACE, mew=1.5,
                 label=f"{name} (ECE {ece:.3f})")
    _style(ax1, "Reliability diagram (OOS)", "Gözlenen Y=1 oranı")
    ax1.set_xlabel("Ortalama tahmin P(Y=1)", color=INK_MUTED, fontsize=9)
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK)
    bins = np.linspace(0, 1, 21)
    for (name, p), color in zip(cal["probs"].items(), SERIES):
        ax2.hist(p.dropna(), bins=bins, histtype="step", lw=2, color=color, label=name)
    ax2.axvline(session.cfg.meta_model.threshold, color=INK, lw=1, ls="--")
    ax2.annotate(f"eşik {session.cfg.meta_model.threshold}", xy=(session.cfg.meta_model.threshold, 0),
                 xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK)
    _style(ax2, "Olasılık histogramı", "Olay sayısı")
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def cost_plot(session: ResearchSession, path: Path) -> Path:
    plt = _plt()
    t = session.results["cost_sensitivity"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    for name, color in list(STRATEGY_COLORS.items())[:3]:
        s = t[t["strategy"] == name]
        ax1.plot(s["cost_bps"], s["Sharpe"], "o-", color=color, lw=2, ms=6, mec=SURFACE, mew=1.5, label=name)
        ax2.plot(s["cost_bps"], s["CAGR"] * 100, "o-", color=color, lw=2, ms=6, mec=SURFACE, mew=1.5, label=name)
    for ax in (ax1, ax2):
        ax.axhline(0, color=INK_MUTED, lw=1, alpha=0.6)
        ax.set_xlabel("Tek yön maliyet (bps)", color=INK_MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    _style(ax1, "Maliyet duyarlılığı — Sharpe", "Sharpe")
    _style(ax2, "Maliyet duyarlılığı — CAGR", "CAGR (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def trade_distribution_plot(session: ResearchSession, path: Path) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE, sharey=False)
    for ax, name in zip(axes, ("Primary", "Meta", "Meta+Sizing")):
        t = session._runs[name].trades * 100
        if t.empty:
            continue
        bins = np.linspace(t.min(), t.max(), 30)
        ax.hist(t[t > 0], bins=bins, color=WIN, alpha=0.85, label=f"Kazanç ({(t > 0).sum()})",
                edgecolor=SURFACE, linewidth=1)
        ax.hist(t[t <= 0], bins=bins, color=LOSS, alpha=0.85, label=f"Kayıp ({(t <= 0).sum()})",
                edgecolor=SURFACE, linewidth=1)
        ax.axvline(t.mean(), color=INK, lw=1, ls="--")
        _style(ax, f"{name}: işlem getirileri", "İşlem sayısı")
        ax.set_xlabel("Net getiri (%)", color=INK_MUTED, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def distribution_plot(session: ResearchSession, path: Path) -> Path:
    """CPCV yol dağılımları ve placebo dağılımları (gerçek Sharpe ile)."""
    plt = _plt()
    cp = session.results["cpcv"]
    dists = session.results["placebo_dists"]
    real = session._runs["Meta"].metrics["Sharpe"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    for i, (name, color) in enumerate((("Meta", SERIES[1]), ("Meta+Sizing", SERIES[2]))):
        v = cp.loc[cp["strategy"] == name, "Sharpe"].to_numpy()
        ax1.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=64, color=color,
                    edgecolor=SURFACE, linewidth=2, zorder=3, label=f"{name} (medyan {np.nanmedian(v):.2f})")
    ax1.axhline(0, color=INK_MUTED, lw=1)
    ax1.axhline(real, color=INK, lw=1, ls="--")
    ax1.annotate("walk-forward Meta Sharpe", xy=(1.3, real), fontsize=8, color=INK, va="bottom")
    ax1.set_xticks([0, 1], ["Meta", "Meta+Sizing"])
    ax1.set_xlim(-0.5, 1.9)
    _style(ax1, f"CPCV: {session.results['cpcv_info']['n_paths']} backtest yolunun Sharpe dağılımı", "Sharpe")
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower left")
    names = sorted(dists)
    for i, (name, color) in enumerate(zip(names, SERIES)):
        v = np.asarray(dists[name], dtype=float)
        v = v[np.isfinite(v)]
        ax2.scatter(v, np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)) if len(v) else [], s=40,
                    color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax2.axvline(real, color=INK, lw=1.5, ls="--")
    ax2.annotate(f"gerçek Sharpe {real:.2f}", xy=(real, len(names) - 0.55), xytext=(-4, 0),
                 textcoords="offset points", fontsize=8, color=INK, ha="right")
    ax2.set_yticks(range(len(names)), names, fontsize=8)
    _style(ax2, "Placebo testleri: Meta Sharpe dağılımı", None)
    ax2.set_xlabel("Sharpe", color=INK_MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return path


def make_all(session: ResearchSession, fig_dir: Path) -> dict[str, Path]:
    try:
        _plt()
    except ImportError:
        session.log.warning("matplotlib yok; grafikler atlandı (pip install matplotlib)")
        return {}
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = {"equity": equity_and_drawdown(session, fig_dir / "equity_drawdown.png")}
    if "calibration" in session.results:
        out["calibration"] = calibration_plot(session, fig_dir / "calibration.png")
    if "cost_sensitivity" in session.results:
        out["costs"] = cost_plot(session, fig_dir / "cost_sensitivity.png")
    out["trades"] = trade_distribution_plot(session, fig_dir / "trade_distribution.png")
    if "cpcv" in session.results and "placebo_dists" in session.results:
        out["distributions"] = distribution_plot(session, fig_dir / "cpcv_placebo.png")
    return out
