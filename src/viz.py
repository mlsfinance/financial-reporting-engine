"""
this is the visualization engine, it generates standardized figures from the prices, returns and KPI outputs

figures:
    * Price line chart
    * Drawdown chart
    * Rolling returns charts (per window)
    * Rolling volatility charts (per window)
    * Return histograms (per asset)
    * Return boxplot (all assets)

the outputs are saved to disk (uniform naming) and paths are returned for reporting
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from .config import Config
from .kpis import KPIResult


# --- style constants ---
_NAVY       = "#1B3A6B"
_ACCENT     = "#2E6FC0"
_MUTED      = "#6B7C99"
_GRID       = "#E0E6EF"
_BG         = "#FFFFFF"
_SPINE      = "#C8D0DC"

# colour palette for multiple assets (navy first, then progressively lighter/warmer)
_PALETTE = ["#1B3A6B", "#2E6FC0", "#E07B39", "#4CAF7D", "#9B59B6", "#E74C3C"]


def _apply_style(ax: plt.Axes, ylabel: str = "") -> None:
    """apply consistent financial-classic styling to an axes object"""
    ax.set_facecolor(_BG)
    ax.figure.patch.set_facecolor(_BG)

    ax.yaxis.set_label_text(ylabel, fontsize=9, color=_MUTED)
    ax.xaxis.set_label_text(ax.get_xlabel(), fontsize=9, color=_MUTED)

    ax.tick_params(axis="both", labelsize=8, colors=_MUTED, length=3)
    ax.grid(True, color=_GRID, linewidth=0.6, linestyle="-", zorder=0)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE)
        spine.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_linewidth(0.6)
        legend.get_frame().set_edgecolor(_SPINE)
        legend.get_frame().set_facecolor(_BG)
        for text in legend.get_texts():
            text.set_fontsize(8)
            text.set_color(_MUTED)
        if legend.get_title():
            legend.get_title().set_fontsize(8)
            legend.get_title().set_color(_MUTED)


def _line_colors(n: int) -> list[str]:
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


class VizError(Exception):
    """ this error is raised when figure generation fails """

@dataclass
class FigureBundle:
    """ collection of saved figure paths (for PDF/HTML reporting) """
    prices: Optional[Path]
    drawdowns: Optional[Path]
    rolling_returns: Dict[int, Path]
    rolling_volatility: Dict[int, Path]
    histograms: Dict[str, Path]
    boxplot: Optional[Path]



def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def _safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_")



def _save_fig(fig: plt.Figure, output_path: Path, dpi: int) -> Path:
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    return output_path



def _get_output_dir(cfg: Config) -> Path:
    out = cfg.get("report.output_pdf", default="./reports/report.pdf", type_=str) or "./reports/report.pdf"
    base = Path(out).resolve().parent
    fig_dir = base / "figures"
    _ensure_dir(fig_dir)
    return fig_dir



def _get_format_and_dpi(cfg: Config) -> tuple[str, int]:
    fmt = cfg.get("visuals.figure_format", default="png", type_=str) or "png"
    dpi = int(cfg.get("visuals.dpi", default=150) or 150)
    return fmt.lower(), dpi


def plot_prices(prices: pd.DataFrame, cfg: Config, title: str = "Prices") -> Path:
    fig_dir = _get_output_dir(cfg)
    fmt, dpi = _get_format_and_dpi(cfg)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = _line_colors(len(prices.columns))
    for i, col in enumerate(prices.columns):
        ax.plot(prices.index, prices[col], linewidth=1.2, color=colors[i], label=str(col))

    base = cfg.get("data.base_currency", default=None, type_=str)
    ylabel = f"Price ({base})" if base else "Price"
    ax.set_xlabel("Date")
    ax.legend(title="Asset", loc="upper left")
    _apply_style(ax, ylabel=ylabel)

    out = fig_dir / f"prices.{fmt}"
    return _save_fig(fig, out, dpi=dpi)


def plot_drawdowns(drawdowns: pd.DataFrame, cfg: Config, title: str = "Drawdowns") -> Path:
    fig_dir = _get_output_dir(cfg)
    fmt, dpi = _get_format_and_dpi(cfg)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = _line_colors(len(drawdowns.columns))
    for i, col in enumerate(drawdowns.columns):
        ax.plot(drawdowns.index, (drawdowns[col] * 100.0), linewidth=1.0, color=colors[i], label=str(col))

    ax.fill_between(drawdowns.index, (drawdowns.min(axis=1) * 100.0), 0,
                    alpha=0.06, color=_NAVY)
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(title="Asset", loc="lower left")
    _apply_style(ax, ylabel="Drawdown (%)")

    out = fig_dir / f"drawdowns.{fmt}"
    return _save_fig(fig, out, dpi=dpi)



def plot_rolling_metric(
    metric: pd.DataFrame,
    window: int,
    cfg: Config,
    title: str,
    ylabel: str,
    filename_prefix: str,
) -> Path:

    fig_dir = _get_output_dir(cfg)
    fmt, dpi = _get_format_and_dpi(cfg)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = _line_colors(len(metric.columns))
    for i, col in enumerate(metric.columns):
        ax.plot(metric.index, metric[col], linewidth=1.0, color=colors[i], label=str(col), alpha=0.85)

    ax.set_xlabel("Date")
    ax.legend(title="Asset", loc="upper left", fontsize=7)
    _apply_style(ax, ylabel=ylabel)

    out = fig_dir / f"{filename_prefix}_{window}.{fmt}"
    return _save_fig(fig, out, dpi=dpi)



def plot_histograms(returns: pd.DataFrame, cfg: Config, bins: int = 50) -> Dict[str, Path]:
    fig_dir = _get_output_dir(cfg)
    fmt, dpi = _get_format_and_dpi(cfg)

    paths: Dict[str, Path] = {}
    for i, col in enumerate(returns.columns):
        series = returns[col].dropna()
        fig, ax = plt.subplots(figsize=(7, 3.5))

        color = _PALETTE[i % len(_PALETTE)]
        ax.hist(series.values, bins=bins, color=color, alpha=0.80, edgecolor=_BG, linewidth=0.4)

        # add a subtle vertical line at zero
        ax.axvline(0, color=_MUTED, linewidth=0.8, linestyle="--", alpha=0.6)

        ax.set_xlabel("Daily Return")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
        _apply_style(ax, ylabel="Frequency")

        out = fig_dir / f"hist_{_safe_stem(str(col))}.{fmt}"
        paths[str(col)] = _save_fig(fig, out, dpi=dpi)

    return paths



def plot_boxplot(returns: pd.DataFrame, cfg: Config, title: str = "Return Boxplot") -> Path:
    fig_dir = _get_output_dir(cfg)
    fmt, dpi = _get_format_and_dpi(cfg)

    fig, ax = plt.subplots(figsize=(7, 4))
    cols = list(returns.columns)
    data = [returns[c].dropna().values for c in cols]
    colors = _line_colors(len(cols))

    bp = ax.boxplot(
        data,
        labels=[str(c) for c in cols],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color=_ACCENT, linewidth=1.5),
        whiskerprops=dict(color=_MUTED, linewidth=0.9),
        capprops=dict(color=_MUTED, linewidth=0.9),
        flierprops=dict(marker="o", markersize=2.5, linestyle="none",
                        markerfacecolor=_MUTED, markeredgecolor=_MUTED, alpha=0.4),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.0)

    ax.axhline(0, color=_MUTED, linewidth=0.7, linestyle="--", alpha=0.5)
    ax.set_xlabel("Asset")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    _apply_style(ax, ylabel="Daily Return")

    out = fig_dir / f"boxplot.{fmt}"
    return _save_fig(fig, out, dpi=dpi)


def generate_all_figures(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    kpi: KPIResult,
    cfg: Config,
) -> FigureBundle:
    """
    this function orchestrates figure creation based on report flags and reads:
      * report.include_drawdowns
      * report.include_rolling
      * report.include_histograms
      * report.include_boxplots
    """
    include_drawdowns = bool(cfg.get("report.include_drawdowns", default=True))
    include_rolling = bool(cfg.get("report.include_rolling", default=True))
    include_histograms = bool(cfg.get("report.include_histograms", default=True))
    include_boxplots = bool(cfg.get("report.include_boxplots", default=True))


    #always generate prices (core chart)
    prices_path = plot_prices(prices, cfg, title="Prices")


    dd_path: Optional[Path] = None
    if include_drawdowns:
        dd_path = plot_drawdowns(kpi.drawdowns, cfg, title="Drawdowns")


    roll_ret_paths: Dict[int, Path] = {}
    roll_vol_paths: Dict[int, Path] = {}


    if include_rolling:
        for w, df in kpi.rolling_returns.items():
            roll_ret_paths[w] = plot_rolling_metric(
                df,
                window=w,
                cfg=cfg,
                title="Rolling Cumulative Return",
                ylabel="Return",
                filename_prefix="rolling_return",
            )
        for w, df in kpi.rolling_volatility.items():
            roll_vol_paths[w] = plot_rolling_metric(
                df,
                window=w,
                cfg=cfg,
                title="Rolling Volatility (Annualized)",
                ylabel="Volatility",
                filename_prefix="rolling_vol",
            )


    hist_paths: Dict[str, Path] = {}
    if include_histograms:
        hist_paths = plot_histograms(returns, cfg, bins=50)


    boxplot_path: Optional[Path] = None
    if include_boxplots and returns.shape[1] >= 1:
        boxplot_path = plot_boxplot(returns, cfg, title="Return Boxplot")


    return FigureBundle(
        prices=prices_path,
        drawdowns=dd_path,
        rolling_returns=roll_ret_paths,
        rolling_volatility=roll_vol_paths,
        histograms=hist_paths,
        boxplot=boxplot_path,
    )