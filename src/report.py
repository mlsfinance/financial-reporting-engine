"""
this is the reporting module. it generates both an HTML and a PDF report

each report will include:
    * basic metadata (title, generation date)
    * a KPI summary table
    * links/embeds to figures (prices, drawdowns, rolling metrics, histograms, boxplot)
    * simple template-based executive summary text
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.platypus import KeepTogether
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader


from .config import Config
from .kpis import KPIResult
from .viz import FigureBundle
from .attribution import AttributionResult
from .messages import RunMessages


class ReportError(Exception):
    """this error is raised when the report generation fails"""


@dataclass
class ReportPaths:
    html: Optional[Path]
    pdf: Optional[Path]


#ensure parents exist
def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _relative_fig_path(fig_path: Path, report_path: Path) -> str:
    """return a figure path relative to the report location for HTML"""
    try:
        return str(fig_path.relative_to(report_path.parent))
    except ValueError:
        #in case of fallback, just use the filename
        return fig_path.name


#generation of the executive summary
def generate_executive_summary(kpi: KPIResult) -> str:
    """
    this is a very simple executive summary, it looks at:
        * best/worst CAGR
        * best/worst Sharpe
        * max drawdowns
    """
    summary = kpi.summary.copy()    


    #guard against empty
    if summary.empty:
        return "Not enough data to compute a meaningful executive summary."


    text_parts: list[str] = []


    #an introduction
    text_parts.append("This report summarizes the recent performance and risk profile of the analyzed asset(s).")


    #CAGR
    if "CAGR" in summary.columns:
        best_cagr_asset = summary["CAGR"].idxmax()
        best_cagr = summary["CAGR"].max()
        worst_cagr_asset = summary["CAGR"].idxmin()
        worst_cagr = summary["CAGR"].min()
        text_parts.append(
            f"The highest compound annual growth rate (CAGR) was observed for '{best_cagr_asset}' "
            f"at approximately {best_cagr:.2%}, while the lowest CAGR was for '{worst_cagr_asset}' "
            f"at around {worst_cagr:.2%}."
        )


    #sharpe
    if "Sharpe" in summary.columns:
        best_sharpe_asset = summary["Sharpe"].idxmax()
        best_sharpe = summary["Sharpe"].max()
        text_parts.append(
            f"In risk-adjusted terms (Sharpe ratio), '{best_sharpe_asset}' showed the most attractive "
            f"profile with a Sharpe ratio of about {best_sharpe:.2f}."
        )


    #drawdowns
    if "Max.Drawdown" in summary.columns:
        worst_dd_asset = summary["Max.Drawdown"].idxmin()
        worst_dd = summary["Max.Drawdown"].min()
        text_parts.append(
            f"The most severe maximum drawdown was recorded for '{worst_dd_asset}', "
            f"with a peak-to-trough loss of approximately {worst_dd:.2%}."
        )


    text_parts.append(
        "These metrics should be interpreted in the context of the investor's risk tolerance, "
        "investment horizon, and diversification objectives."
    )


    return " ".join(text_parts)

def _get_portfolio_weights_from_attrib(attrib: Optional[AttributionResult]) -> Optional[pd.Series]:
    if attrib is None:
        return None
    if attrib.table is None or attrib.table.empty:
        return None
    if "Weight" not in attrib.table.columns:
        return None
    w = attrib.table["Weight"].copy()
    w.index = w.index.astype(str)
    w = w.astype(float)
    s = float(w.sum())
    if s != 0.0:
        w = w / s
    return w


def _portfolio_returns(returns: pd.DataFrame, weights: Optional[pd.Series]) -> pd.Series:
    r = returns.copy()
    r.index = pd.to_datetime(r.index)
    r = r.dropna(how="all")
    if r.empty:
        return pd.Series(dtype=float)

    if weights is None:
        return r.mean(axis=1)

    w = weights.reindex(r.columns).fillna(0.0).astype(float)
    s = float(w.sum())
    if s == 0.0:
        return r.mean(axis=1)
    w = w / s

    return (r.mul(w, axis=1)).sum(axis=1)


def _compound_return(r: pd.Series) -> float:
    s = r.dropna()
    if len(s) < 2:
        return float("nan")
    return float((1.0 + s).prod() - 1.0)



def compute_headline_metrics(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    attrib: Optional[AttributionResult],
    cfg: Config,
) -> dict:
    """
    these are the headline metrics for first page of the report. the function uses weighted portfolio returns if attrib is available, if not then equal-weight.
    """
    period_start_str = cfg.get("portfolio.return_period_start", default=None, type_=str)
    as_of_str = cfg.get("portfolio.as_of", default=None, type_=str)

    as_of = pd.to_datetime(as_of_str) if as_of_str else None
    period_start = pd.to_datetime(period_start_str) if period_start_str else None

    w = _get_portfolio_weights_from_attrib(attrib)
    port_r = _portfolio_returns(returns, w)

    if as_of is not None:
        port_r = port_r.loc[:as_of]

    # period return
    period_slice = port_r
    if period_start is not None:
        period_slice = period_slice.loc[period_start:]
    if as_of is not None:
        period_slice = period_slice.loc[:as_of]
    period_ret = _compound_return(period_slice)

    # YTD return
    if port_r.dropna().empty:
        ytd_ret = float("nan")
    else:
        last_date = port_r.dropna().index[-1]
        ytd_start = pd.Timestamp(year=last_date.year, month=1, day=1)
        ytd_slice = port_r.loc[ytd_start:]
        if as_of is not None:
            ytd_slice = ytd_slice.loc[:as_of]
        ytd_ret = _compound_return(ytd_slice)

    # annualized vol
    ann_factor = int(cfg.get("kpis.annualization_factor", default=252) or 252)
    pr = port_r.dropna()
    ann_vol = float(pr.std() * (ann_factor ** 0.5)) if len(pr) >= 2 else float("nan")

    # max drawdown
    if len(pr) >= 2:
        cum = (1.0 + pr).cumprod()
        peak = cum.cummax()
        dd = (cum / peak) - 1.0
        max_dd = float(dd.min())
    else:
        max_dd = float("nan")

    return {
        "period_return": period_ret,
        "ytd_return": ytd_ret,
        "annualized_vol": ann_vol,
        "max_drawdown": max_dd,
    }



def compute_simple_drivers(prices: pd.DataFrame, returns: pd.DataFrame, kpi: KPIResult, attrib: Optional[AttributionResult], cfg: Config) -> list[str]:
    """
    3 simple rules-based drivers:
      1) best / worst asset period performance
      2) volatility regime (last window vs longer window if available)
      3) drawdown severity flag
    """
    drivers: list[str] = []

    if attrib is not None and not attrib.top3.empty:
        tops = ", ".join(attrib.top3.index.tolist())
        bottoms = ", ".join(attrib.bottom3.index.tolist())
        drivers.append(f"Top contributors: {tops}. Main detractors: {bottoms}.")


    #1) period performance by asset (approximately from returns)
    r = returns.copy().dropna(how="all")
    if not r.empty:
        period_asset = (1.0 + r).prod(axis=0) - 1.0
        best = period_asset.idxmax()
        worst = period_asset.idxmin()
        drivers.append(f"Performance dispersion: '{best}' led the period, while '{worst}' lagged.")

    #2) Volatility regime (use rolling if present). if the kpi object contains rolling_volatility, it compares the largest vs the smallest window
    try:
        rv = kpi.rolling_volatility  #dict[int, DataFrame]
        if rv:
            windows = sorted(rv.keys())
            w_short, w_long = windows[0], windows[-1]
            
            #portfolio proxy: mean across assets
            short_last = rv[w_short].mean(axis=1).dropna().iloc[-1]
            long_last = rv[w_long].mean(axis=1).dropna().iloc[-1]
            if short_last > long_last:
                drivers.append(f"Volatility increased recently (short window {w_short} > long window {w_long}).")
            else:
                drivers.append(f"Volatility eased recently (short window {w_short} ≤ long window {w_long}).")
    except Exception:
        pass

    #3) drawdown flag on portfolio proxy
    hm = compute_headline_metrics(prices, returns, attrib, cfg)
    max_dd = hm.get("max_drawdown", float("nan"))
    if pd.notna(max_dd):
        if max_dd <= -0.20:
            drivers.append("Drawdown pressure was material (max drawdown below -20%).")
        elif max_dd <= -0.10:
            drivers.append("Drawdown remained noticeable (max drawdown below -10%).")
        else:
            drivers.append("Drawdown stayed contained (max drawdown above -10%).")

    #it ensures there're exactly 3 bullets
    return drivers[:3] if len(drivers) >= 3 else drivers + [""] * (3 - len(drivers))


#the HTML report
def _build_html(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    kpi: KPIResult,
    figs: FigureBundle,
    attrib: Optional[AttributionResult],
    cfg: Config,
    msgs: RunMessages | None = None
) -> Optional[Path]:
    output_html_str = cfg.get("report.output_html", default="./reports/report.html", type_=str)


    if not output_html_str:
        return None


    output_html = Path(output_html_str).resolve()
    _ensure_parent_dir(output_html)


    title = cfg.get("report.title", default="Financial Report", type_=str) or "Financial Report"
    generated_at = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")


    exec_summary = generate_executive_summary(kpi)

    hm = compute_headline_metrics(prices, returns, attrib, cfg)
    drivers = compute_simple_drivers(prices, returns, kpi, attrib, cfg)

    attrib_html = ""
    if attrib is not None:
        def _fmt(x):
            return f"{x:.2%}"

        rows = []
        for label, df_ in [("Top contributors", attrib.top3), ("Bottom detractors", attrib.bottom3)]:
            rows.append(f"<h3>{label}</h3>")
            rows.append("<table class='attrib-table'>")
            rows.append("<tr><th>Asset</th><th>Weight</th><th>Return</th><th>Contribution</th></tr>")
            for asset, r in df_.iterrows():
                rows.append(
                    f"<tr>"
                    f"<td>{asset}</td>"
                    f"<td>{_fmt(r['Weight'])}</td>"
                    f"<td>{_fmt(r['Period Return'])}</td>"
                    f"<td>{_fmt(r['Contribution'])}</td>"
                    f"</tr>"
                )
            rows.append("</table>")

        attrib_html = "\n".join(rows)

    warnings_html = ""
    if msgs and msgs.warnings:
        items = "\n".join(f"<li>{w}</li>" for w in msgs.warnings)
        warnings_html = f"""
        <div class="warn-box">
        <div class="warn-title">Data quality notes</div>
        <ul>{items}</ul>
        </div>
        """


    headline_html = f"""
    <table class="headline-table-2col">
        <tr><td class="label">Period return</td><td class="value">{hm['period_return']:.2%}</td></tr>
        <tr><td class="label">YTD</td><td class="value">{hm['ytd_return']:.2%}</td></tr>
        <tr><td class="label">Annualized vol</td><td class="value">{hm['annualized_vol']:.2%}</td></tr>
        <tr><td class="label">Max drawdown</td><td class="value">{hm['max_drawdown']:.2%}</td></tr>
    </table>

    <ul class="drivers">
        <li>{drivers[0]}</li>
        <li>{drivers[1]}</li>
        <li>{drivers[2]}</li>
    </ul>
    """




    base_ccy = cfg.get("data.base_currency", default=None, type_=str)
    unit = f" ({base_ccy})" if base_ccy else ""

    #the KPI summary table as HTML
    kpi_table_html = kpi.summary.reset_index().to_html(
        classes="kpi-table",
        index=False,
        border=0,
        float_format=lambda x: f"{x:0.2f}",
    )



    #build image tags if paths already exist
    img_tags = []


    def add_img_tag(fig_path: Optional[Path], caption: str) -> None:
        if fig_path is None:
            return
        rel = _relative_fig_path(fig_path, output_html)
        img_tags.append(
            f'<div class="figure-block"><div class="fig-title">{caption}</div>'
            f'<img src="{rel}" alt="{caption}"></div>'
        )


    add_img_tag(figs.prices, "Price Chart")
    add_img_tag(figs.drawdowns, "Drawdown Chart")


    for w, path in sorted(figs.rolling_returns.items()):
        add_img_tag(path, f"Rolling Cumulative Return (window={w})")


    for w, path in sorted(figs.rolling_volatility.items()):
        add_img_tag(path, f"Rolling Volatility (window={w})")


    for asset, path in figs.histograms.items():
        add_img_tag(path, f"Return Distribution: {asset}")


    add_img_tag(figs.boxplot, "Return Boxplot")


    img_section_html = "\n".join(img_tags)


    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
    :root {{
        --navy:      #1B3A6B;
        --accent:    #2E6FC0;
        --text:      #1A1A2E;
        --muted:     #6B7C99;
        --border:    #D0D9E8;
        --header-bg: #1B3A6B;
        --header-fg: #FFFFFF;
        --row-alt:   #F4F7FC;
        --card-bg:   #F8FAFD;
        --bg:        #FFFFFF;
    }}
    * {{ box-sizing: border-box; }}
    body {{
        font-family: "Georgia", "Times New Roman", serif;
        color: var(--text);
        background: var(--bg);
        margin: 0;
        padding: 0;
        font-size: 13px;
        line-height: 1.55;
    }}
    .page-wrap {{
        max-width: 960px;
        margin: 0 auto;
        padding: 0 32px 48px 32px;
    }}

    /* header bar */
    .report-header {{
        background: var(--navy);
        color: #fff;
        padding: 28px 32px 22px 32px;
        margin-bottom: 32px;
    }}
    .report-header h1 {{
        margin: 0 0 4px 0;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 0.3px;
        color: #fff;
    }}
    .report-header .meta {{
        font-size: 11px;
        color: #A8BDD8;
        font-family: Arial, sans-serif;
        margin: 0;
    }}

    /* section headings */
    h2 {{
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--navy);
        border-bottom: 2px solid var(--navy);
        padding-bottom: 5px;
        margin: 36px 0 16px 0;
        font-family: Arial, sans-serif;
    }}
    h3 {{
        font-size: 12px;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin: 20px 0 8px 0;
        font-family: Arial, sans-serif;
    }}

    /* headline metric cards */
    .metric-cards {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 20px 0 24px 0;
    }}
    .metric-card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-top: 3px solid var(--accent);
        padding: 14px 16px 12px 16px;
        border-radius: 2px;
    }}
    .metric-card .label {{
        font-size: 10px;
        font-family: Arial, sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--muted);
        margin-bottom: 6px;
    }}
    .metric-card .value {{
        font-size: 20px;
        font-weight: 700;
        color: var(--navy);
        font-family: Arial, sans-serif;
        line-height: 1;
    }}
    .metric-card .value.negative {{ color: #C0392B; }}
    .metric-card .value.positive {{ color: #1A7A4A; }}

    /* drivers list */
    .drivers {{
        margin: 0 0 8px 0;
        padding-left: 18px;
        color: var(--text);
    }}
    .drivers li {{
        margin: 5px 0;
        font-family: Arial, sans-serif;
        font-size: 12.5px;
    }}

    /* executive summary */
    .exec-text {{
        font-size: 13px;
        color: #333;
        line-height: 1.65;
        margin: 0 0 8px 0;
    }}

    /* tables */
    table {{
        border-collapse: collapse;
        width: 100%;
        font-family: Arial, sans-serif;
        font-size: 12px;
    }}
    th {{
        background: var(--header-bg);
        color: var(--header-fg);
        font-weight: 600;
        padding: 8px 10px;
        text-align: right;
        border: 1px solid var(--navy);
        letter-spacing: 0.3px;
    }}
    th:first-child {{ text-align: left; }}
    td {{
        padding: 7px 10px;
        border: 1px solid var(--border);
        text-align: right;
    }}
    td:first-child {{ text-align: left; font-weight: 500; }}
    tr:nth-child(even) td {{ background: var(--row-alt); }}

    .kpi-table, .attrib-table {{
        margin: 0 0 18px 0;
    }}

    /* figures */
    .figure-block {{
        margin: 30px 0;
        border: 1px solid var(--border);
        border-radius: 2px;
        overflow: hidden;
    }}
    .figure-block .fig-title {{
        background: var(--card-bg);
        border-bottom: 1px solid var(--border);
        padding: 8px 14px;
        font-size: 11px;
        font-family: Arial, sans-serif;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--navy);
    }}
    .figure-block img {{
        display: block;
        width: 100%;
        height: auto;
    }}

    /* warn box */
    .warn-box {{
        border-left: 4px solid #E07B39;
        background: #FFF8F0;
        padding: 10px 14px;
        margin: 16px 0;
        border-radius: 2px;
        font-family: Arial, sans-serif;
        font-size: 12px;
    }}
    .warn-title {{
        font-weight: 700;
        color: #C05A10;
        margin-bottom: 4px;
    }}

    @media (max-width: 640px) {{
        .metric-cards {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    </style>
</head>
<body>
    <div class="report-header">
        <h1>{title}</h1>
        <p class="meta">Generated on {generated_at}</p>
    </div>
    <div class="page-wrap">

    {warnings_html}

    <h2>Executive Summary</h2>
    <p class="exec-text">{exec_summary}</p>

    <div class="metric-cards">
        <div class="metric-card">
            <div class="label">Period Return</div>
            <div class="value {'positive' if hm['period_return'] >= 0 else 'negative'}">{hm['period_return']:.2%}</div>
        </div>
        <div class="metric-card">
            <div class="label">YTD</div>
            <div class="value {'positive' if hm['ytd_return'] >= 0 else 'negative'}">{hm['ytd_return']:.2%}</div>
        </div>
        <div class="metric-card">
            <div class="label">Annualized Vol</div>
            <div class="value">{hm['annualized_vol']:.2%}</div>
        </div>
        <div class="metric-card">
            <div class="label">Max Drawdown</div>
            <div class="value negative">{hm['max_drawdown']:.2%}</div>
        </div>
    </div>

    <h3>Key Drivers</h3>
    <ul class="drivers">
        <li>{drivers[0]}</li>
        <li>{drivers[1]}</li>
        <li>{drivers[2]}</li>
    </ul>

    {'<h2>Attribution</h2>' + attrib_html if attrib_html else ''}

    <h2>Key Performance Indicators</h2>
    {kpi_table_html}

    <h2>Figures</h2>
    {img_section_html}

    </div>
</body>
</html>
"""


    output_html.write_text(html, encoding="utf-8")
    return output_html
   
   
#the PDF report
def _build_pdf(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    kpi: KPIResult,
    figs: FigureBundle,
    attrib: Optional[AttributionResult],
    cfg: Config,
    msgs: RunMessages | None = None
) -> Optional[Path]:
    
    output_pdf_str = cfg.get("report.output_pdf", default="./reports/report.pdf", type_=str)
    if not output_pdf_str:
        return None


    output_pdf = Path(output_pdf_str).resolve()
    _ensure_parent_dir(output_pdf)


    title = cfg.get("report.title", default="Financial Report", type_=str) or "Financial Report"
    generated_at = datetime.now().strftime("%Y-%m-%d at %H:%M:%S")


    exec_summary = generate_executive_summary(kpi)


    hm = compute_headline_metrics(prices, returns, attrib, cfg)
    drivers = compute_simple_drivers(prices, returns, kpi, attrib, cfg)
    base_ccy = cfg.get("data.base_currency", default=None, type_=str)


    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )


    NAVY_COLOR  = colors.HexColor("#1B3A6B")
    MUTED_COLOR = colors.HexColor("#6B7C99")

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="TitleCenter",
        parent=styles["Title"],
        alignment=0,
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        spaceAfter=2,
        textColor=colors.white,
    ))

    styles.add(ParagraphStyle(
        name="Heading",
        parent=styles["Heading2"],
        alignment=0,
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        spaceBefore=14,
        spaceAfter=5,
        textColor=NAVY_COLOR,
    ))

    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=5,
        textColor=colors.HexColor("#1A1A2E"),
    ))

    styles.add(ParagraphStyle(
        name="Meta",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.white,
        spaceAfter=0,
    ))

    elements = []

    NAVY       = colors.HexColor("#1B3A6B")
    ACCENT     = colors.HexColor("#2E6FC0")
    BORDER     = colors.HexColor("#D0D9E8")
    HEADER_BG  = colors.HexColor("#1B3A6B")
    HEADER_FG  = colors.white
    LABEL_BG   = colors.HexColor("#F4F7FC")
    ROW_ALT    = colors.HexColor("#F4F7FC")

    def apply_table_style(t: Table, header: bool = True) -> None:
        ts = [
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
        ]
        if header:
            if header:
                ts += [
                    ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),  
                ]
        t.setStyle(TableStyle(ts))


    def add_attribution_table(attrib: Optional[AttributionResult]) -> None:
        if attrib is None:
            return

        def pct(x: float) -> str:
            return "" if pd.isna(x) else f"{x:.2%}"

        def row_block(title_txt: str, block: pd.DataFrame) -> list:
            # block index = asset names
            header = Paragraph(title_txt, styles["Heading"])
            table_rows = [["Asset", "Weight", "Return", "Contribution"]]

            for asset, r in block.iterrows():
                table_rows.append([
                    str(asset),
                    pct(float(r["Weight"])),
                    pct(float(r["Period Return"])),
                    pct(float(r["Contribution"])),
                ])

            # width aligned with your charts (table_width=16cm)
            table_width = 16 * cm
            col_widths = [5.0 * cm, 3.5 * cm, 3.5 * cm, 4.0 * cm]

            t = Table(table_rows, colWidths=col_widths, hAlign="CENTER", repeatRows=1)
            apply_table_style(t, header=True)
            t.setStyle(TableStyle([
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]))


            return [header, Spacer(1, 0.15 * cm), t, Spacer(1, 0.45 * cm)]

        top_block = attrib.top3.copy() if attrib.top3 is not None else pd.DataFrame()
        bot_block = attrib.bottom3.copy() if attrib.bottom3 is not None else pd.DataFrame()

        # Guard in case of empty
        if top_block.empty and bot_block.empty:
            return

        blocks = []  # <-- IMPORTANT: always define

        if not top_block.empty:
            blocks.append(KeepTogether(row_block("Top contributors", top_block)))
        if not bot_block.empty:
            blocks.append(KeepTogether(row_block("Bottom detractors", bot_block)))

        # Title kept with first block to avoid orphan title
        elements.append(
            KeepTogether([
                Paragraph("Attribution (Period Contribution)", styles["Heading"]),
                Spacer(1, 0.2 * cm),
                blocks[0],
            ])
        )

        # Append the remaining blocks (if any)
        for b in blocks[1:]:
            elements.append(b)

        elements.append(Spacer(1, 0.4 * cm))


    #title header bar (navy background)
    header_table = Table(
        [[Paragraph(title, styles["TitleCenter"]),
          Paragraph(f"Generated on {generated_at}", styles["Meta"])]],
        colWidths=[13 * cm, 4 * cm],
        hAlign="LEFT",
    )
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1B3A6B")),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("BOX", (0, 0), (-1, -1), 0, colors.HexColor("#1B3A6B")),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 0.5 * cm))

    if msgs and msgs.warnings:
        elements.append(Paragraph("Data quality notes", styles["Heading"]))
        for w in msgs.warnings:
            elements.append(Paragraph(f"• {w}", styles["Body"]))
        elements.append(Spacer(1, 0.3 * cm))

    #headline metrics as 4-column table (like cards)
    CARD_LABEL = colors.HexColor("#F4F7FC")
    CARD_BG    = colors.white
    ACCENT_C   = colors.HexColor("#2E6FC0")

    metric_labels = ["Period Return", "YTD", "Ann. Volatility", "Max Drawdown"]
    metric_values = [
        f"{hm['period_return']:.2%}",
        f"{hm['ytd_return']:.2%}",
        f"{hm['annualized_vol']:.2%}",
        f"{hm['max_drawdown']:.2%}",
    ]

    label_style = ParagraphStyle("CardLabel", fontName="Helvetica", fontSize=8,
                                  textColor=colors.HexColor("#6B7C99"), leading=10)
    val_style   = ParagraphStyle("CardVal", fontName="Helvetica-Bold", fontSize=16,
                                  textColor=colors.HexColor("#1B3A6B"), leading=20)

    card_rows = [
        [Paragraph(l, label_style) for l in metric_labels],
        [Paragraph(v, val_style)   for v in metric_values],
    ]
    card_w = 17.0 / 4 * cm
    m_table = Table(card_rows, colWidths=[card_w] * 4, hAlign="LEFT")
    m_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CARD_BG),
        ("BOX",        (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D9E8")),
        ("INNERGRID",  (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D9E8")),
        ("LINEABOVE",  (0, 0), (-1, 0), 3, ACCENT_C),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))

    elements.append(m_table)
    elements.append(Spacer(1, 0.4 * cm))

    #drivers bullets
    elements.append(Paragraph("Key Drivers", styles["Heading"]))
    for d in drivers:
        if d.strip():
            elements.append(Paragraph(f"• {d}", styles["Body"]))

    elements.append(Spacer(1, 0.4 * cm))
    add_attribution_table(attrib)
    elements.append(Spacer(1, 0.3 * cm))

    #executive Summary
    elements.append(Paragraph("Executive Summary", styles["Heading"]))
    elements.append(Paragraph(exec_summary, styles["Body"]))
    elements.append(Spacer(1, 0.5 * cm))

    df = kpi.summary.copy().reset_index()  #include "Asset" as first column

    #format values for compactness
    def fmt(col: str, x):
        if pd.isna(x):
            return ""
        if col in ("CAGR", "Ann.Vol", "Max.Drawdown"):
            return f"{x:.2%}"
        if col in ("Sharpe", "Sortino"):
            return f"{x:.2f}"
        return f"{x:.2f}" if isinstance(x, (int, float)) else str(x)

    #build rows (as strings)
    columns = list(df.columns)  #["Asset", "CAGR", "Ann.Vol", "Sharpe", "Sortino", "Max.Drawdown"]
    rows = []
    for _, r in df.iterrows():
        row = [str(r["Asset"])]
        for c in columns[1:]:
            row.append(fmt(c, r[c]))
        rows.append(row)

    #pretty header labels (wrap-friendly)
    pretty = {
        "Asset": "Asset",
        "CAGR": "CAGR<br/>(%)",
        "Ann.Vol": "Ann.Vol<br/>(%)",
        "Sharpe": "Sharpe<br/>Ratio",
        "Sortino": "Sortino<br/>Ratio",
        "Max.Drawdown": "Max Drawdown<br/>(%)",
    }

    header_style = ParagraphStyle(
        name="HeaderCell",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=9,
        alignment=1,  #center
        textColor=colors.white,
    )

    cell_style = ParagraphStyle(
        name="Cell",
        parent=styles["BodyText"],
        fontSize=9,
        leading=10,
    )

    header_row = [Paragraph(pretty.get(c, c), header_style) for c in columns]

    table_data = [header_row] + rows

    #fixed width to match charts
    table_width = 16 * cm
    asset_w = 3.0 * cm
    other_w = (table_width - asset_w) / (len(columns) - 1)
    col_widths = [asset_w] + [other_w] * (len(columns) - 1)

    kpi_table = Table(table_data, colWidths=col_widths, hAlign="CENTER", repeatRows=1)

    apply_table_style(kpi_table, header=True)
    kpi_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))


    elements.append(
        KeepTogether([
            Paragraph("Key Performance Indicators", styles["Heading"]),
            Spacer(1, 0.2 * cm),
            kpi_table,
            Spacer(1, 0.7 * cm),
        ])
    )


    def add_image(path: Optional[Path], caption: str) -> None:
        if path is None or not path.exists():
            return

        max_width = 16 * cm
        max_height = 9.5 * cm

        ir = ImageReader(str(path))
        iw, ih = ir.getSize()
        scale = min(max_width / iw, max_height / ih)

        img = RLImage(str(path), width=iw * scale, height=ih * scale)

        elements.append(
            KeepTogether([
                Paragraph(caption, styles["Heading"]),
                Spacer(1, 0.2 * cm),
                img,
                Spacer(1, 1.3 * cm),
            ])
        )


    #figures
    add_image(figs.prices, "Price Chart")
    add_image(figs.drawdowns, "Drawdown Chart")


    for w, path in sorted(figs.rolling_returns.items()):
        add_image(path, f"Rolling Cumulative Return (window={w})")


    for w, path in sorted(figs.rolling_volatility.items()):
        add_image(path, f"Rolling Volatility (window={w})")


    for asset, path in figs.histograms.items():
        add_image(path, f"Return Distribution: {asset}")


    add_image(figs.boxplot, "Return Boxplot")


    doc.build(elements)
    return output_pdf



#the public entrypoint
def generate_reports(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    kpi: KPIResult,
    figs: FigureBundle,
    attrib: Optional[AttributionResult],
    cfg: Config,
    msgs: RunMessages | None = None
) -> ReportPaths:
    
    html_path = _build_html(prices, returns, kpi, figs, attrib, cfg, msgs=msgs)
    pdf_path = _build_pdf(prices, returns, kpi, figs, attrib, cfg, msgs=msgs)
    return ReportPaths(html=html_path, pdf=pdf_path)