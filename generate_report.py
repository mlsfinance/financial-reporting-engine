"""
CLI entrypoint for Financial Data Automation & Reporting Engine

usage: python generate_report.py --config config/example.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from src.cleaner import DataCleanerError, clean_and_normalize
from src.config import Config, ConfigError
from src.kpis import KPIError, compute_kpis
from src.loader import DataLoaderError, load_prices
from src.report import ReportError, generate_reports
from src.viz import VizError, generate_all_figures
from src.attribution import compute_attribution
from src.messages import RunMessages

msgs = RunMessages()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate financial reports from YAML config.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file (e.g. config/example.yaml)",
    )
    return parser.parse_args()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config).resolve()

    try:
        cfg = Config.from_file(cfg_path)

        #create run directory
        run_ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        runs_root = Path(cfg.get("report.output_dir", default="./reports/runs", type_=str) or "./reports/runs")
        run_dir = (runs_root / run_ts).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        #override report outputs to point to run dir
        cfg.set("report.output_pdf", str(run_dir / "report.pdf"))
        cfg.set("report.output_html", str(run_dir / "report.html"))

        #save used config for traceability
        shutil.copy(cfg_path, run_dir / "config_used.yaml")

        #1) load
        loaded = load_prices(cfg, msgs=msgs)
        prices = loaded.prices
        weights = loaded.weights

        #2) clean + normalize + returns
        cleaned = clean_and_normalize(prices, cfg)
        prices_clean = cleaned.prices
        returns = cleaned.returns
        attrib = compute_attribution(returns, weights) if weights is not None else None


        #3) KPIs
        kpi = compute_kpis(returns, cfg)

        #4) visuals
        figs = generate_all_figures(prices_clean, returns, kpi, cfg)

        #5) reports
        outputs = generate_reports(prices_clean, returns, kpi, figs, attrib, cfg, msgs=msgs)

        #run metadata
        metadata: dict = {
            "timestamp": run_ts,
            "config_path": str(cfg_path),
            "run_dir": str(run_dir),
            "assets": [str(c) for c in prices_clean.columns],
            "base_currency": cfg.get("data.base_currency", default=None, type_=str),
        }

        input_path = cfg.get("data.input_path", default=None, type_=str)
        if input_path:
            p = Path(input_path).resolve()
            if p.exists():
                metadata["input_file"] = str(p)
                metadata["input_file_hash_sha256"] = _hash_file(p)

        (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        print("Report generation completed.")
        if outputs.pdf:
            print(f"PDF:  {outputs.pdf}")
        if outputs.html:
            print(f"HTML: {outputs.html}")
        print(f"RUN:  {run_dir}")

        return 0

    except (ConfigError, DataLoaderError, DataCleanerError, KPIError, VizError, ReportError) as e:
        print(f"[ERROR] {e}")
        return 1

    except Exception:  # noqa: BLE001
        import traceback

        print("[UNEXPECTED ERROR]")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
