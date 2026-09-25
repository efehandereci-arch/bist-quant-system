"""Komut satırı: ``python -m meta_labeling [--csv VERI.csv] [--primary ema|bollinger] ...``"""

from __future__ import annotations

import argparse
from dataclasses import replace

from .config import PipelineConfig
from .data import load_ohlcv_csv
from .pipeline import MetaLabelingPipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Meta-Labeling + Triple Barrier pipeline")
    parser.add_argument("--csv", help="OHLCV CSV dosyası (Date,Open,High,Low,Close,Volume). Yoksa simülasyon.")
    parser.add_argument("--primary", choices=["ema", "bollinger"], default=None)
    parser.add_argument("--events", choices=["cusum", "signal"], default=None, help="Olay örnekleme modu")
    parser.add_argument("--model", choices=["lgbm", "rf"], default=None)
    parser.add_argument("--threshold", type=float, default=None, help="P(Y=1) eşiği")
    parser.add_argument("--pt", type=float, default=None, help="Kâr al katsayısı (x volatilite)")
    parser.add_argument("--sl", type=float, default=None, help="Zarar kes katsayısı (x volatilite)")
    parser.add_argument("--horizon", type=int, default=None, help="Dikey bariyer (bar)")
    parser.add_argument("--n-bars", type=int, default=None, help="Simülasyon bar sayısı")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = PipelineConfig()

    def upd(obj, **kw):
        kw = {k: v for k, v in kw.items() if v is not None}
        return replace(obj, **kw) if kw else obj

    cfg = replace(
        cfg,
        sim=upd(cfg.sim, n_bars=args.n_bars, seed=args.seed),
        primary=upd(cfg.primary, kind=args.primary, event_mode=args.events),
        barrier=upd(cfg.barrier, pt_mult=args.pt, sl_mult=args.sl, max_holding_bars=args.horizon),
        model=upd(cfg.model, kind=args.model, threshold=args.threshold, seed=args.seed),
    )
    ohlcv = load_ohlcv_csv(args.csv) if args.csv else None
    result = MetaLabelingPipeline(cfg).run(ohlcv)
    print(result.summary())


if __name__ == "__main__":
    main()
