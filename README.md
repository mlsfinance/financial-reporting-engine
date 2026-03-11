# Financial Data Automation & Reporting Engine
A Python-based CLI pipeline that transforms raw financial data into structured, professional-grade reports — covering portfolio performance, risk metrics, and automated visualizations.

Built as part of a Bachelor's thesis in International Business with a focus on Quantitative Finance.

---

## What it does

Given a YAML configuration file, the engine:

1. **Loads** price data from CSV, Excel, or Yahoo Finance (via `yfinance`)
2. **Cleans & normalizes** the data — handles missing values, duplicates, timezone alignment, and resampling
3. **Computes KPIs** — CAGR, annualized volatility, Sharpe ratio, Sortino ratio, max drawdown, rolling metrics
4. **Runs attribution analysis** — contribution per asset weighted by portfolio position
5. **Generates visualizations** — price charts, drawdown charts, rolling return/volatility charts, return distributions, boxplots
6. **Outputs a report** in both **PDF and HTML** with an executive summary, KPI tables, and all figures

Each run is fully reproducible: the used config and a SHA-256 hash of the input file are saved alongside the report.

---

## Example output

> Sample report available in [`sample_output/`](sample_output/)

---

## Project structure
```
├── generate_report.py       # CLI entrypoint
├── config/
│   ├── example.yaml         # Annotated template with all options
│   ├── csv.yaml             # CSV data source preset
│   ├── excel.yaml           # Excel data source preset
│   └── yfinance.yaml        # Yahoo Finance multi-ticker preset
├── src/
│   ├── config.py            # YAML config loader with dot-notation access
│   ├── loader.py            # Data ingestion (CSV / Excel / yfinance)
│   ├── cleaner.py           # Cleaning, normalization, return calculation
│   ├── kpis.py              # KPI computation (CAGR, Sharpe, drawdowns, rolling)
│   ├── attribution.py       # Portfolio contribution analysis
│   ├── viz.py               # Figure generation (matplotlib)
│   ├── report.py            # PDF & HTML report assembly
│   └── messages.py          # Run-time warnings and info messages
└── sample_output/           # Example report output
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure your run

Edit one of the provided YAML presets in `config/`, or use `config/example.yaml` as a reference — it documents every available option.

**Example: pull data from Yahoo Finance**
```yaml
data:
  source_type: "yfinance"
  tickers: ["AAPL", "GOOGL", "BBVA.MC"]
  start_date: "2022-01-01"
  end_date: null
  base_currency: "USD"

kpis:
  annualization_factor: 252
  risk_free_rate: 0.0
  rolling_windows: [21, 63, 126, 252]

report:
  title: "My Portfolio Report"
  output_pdf: "./reports/report.pdf"
  output_html: "./reports/report.html"
```

### 3. Run
```bash
python generate_report.py --config config/yfinance.yaml
```

Output:
```
Report generation completed.
PDF:  ./reports/runs/2026-01-13_115657/report.pdf
HTML: ./reports/runs/2026-01-13_115657/report.html
RUN:  ./reports/runs/2026-01-13_115657/
```

---

## Key metrics computed

| Metric | Description |
|---|---|
| CAGR | Compound Annual Growth Rate |
| Annualized Volatility | Annualized standard deviation of returns |
| Sharpe Ratio | Risk-adjusted return (vs. configurable risk-free rate) |
| Sortino Ratio | Downside-risk-adjusted return |
| Max Drawdown | Largest peak-to-trough decline |
| Rolling Returns | Cumulative return over configurable windows |
| Rolling Volatility | Annualized vol over configurable windows |
| Attribution | Per-asset contribution to portfolio return |

---

## Configuration reference

See [`config/example.yaml`](config/example.yaml) for a fully annotated template covering all options across `data`, `cleaning`, `kpis`, `visuals`, and `report` sections.

---

## Tech stack

- **Python** — Pandas, NumPy, Matplotlib
- **Data sources** — CSV, Excel (`openpyxl`), Yahoo Finance (`yfinance`)
- **Reporting** — ReportLab (PDF), HTML
- **Config** — PyYAML
- **CLI** — argparse

