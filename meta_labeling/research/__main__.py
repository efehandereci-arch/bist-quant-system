"""``python -m meta_labeling.research --config config.yaml [--ticker SASA] [--universe]``"""

from __future__ import annotations

import argparse

import pandas as pd

from .session import ResearchSession, run_universe
from .settings import load_research_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Meta-labeling research framework")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--ticker", default=None, help="Varsayılan: config'deki ilk hisse")
    parser.add_argument("--universe", action="store_true", help="Tüm config hisselerinde çekirdek OOS testi")
    args = parser.parse_args(argv)
    cfg = load_research_config(args.config)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        if args.universe:
            print(run_universe(cfg).round(3).to_string())
            return
        report = ResearchSession(cfg, args.ticker).run_all()
        print(report.final_table.to_string())
        print()
        print(report.robustness_table.to_string(index=False))
        print(f"\nKARAR: {report.verdict}\nRapor: {report.path}")


if __name__ == "__main__":
    main()
