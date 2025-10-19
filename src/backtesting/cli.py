from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Optional

from src.backtesting.engine import logical_test, quality_test
from src.backtesting.loader import list_dates
from src.utils.logbook_utils import resolve_logbook_dir


def _parse_dates_arg(
    arg: Optional[str], base_dir: str, table: str, symbol: str
) -> Optional[list[str]]:
    if not arg:
        return None
    if arg == "all":
        return list_dates(base_dir, table, symbol)
    return [s.strip() for s in arg.split(",") if s.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("backtest")
    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("logical", help="Run logical test over a small slice")
    lp.add_argument("symbol")
    lp.add_argument(
        "--dates", help="Comma-separated YYYY-MM-DD list or 'all'", default=None
    )
    lp.add_argument("--max-files", type=int, default=10)
    lp.add_argument("--logbook-dir", default=None)

    qp = sub.add_parser("quality", help="Run quality test over extended data")
    qp.add_argument("symbol")
    qp.add_argument(
        "--dates", help="Comma-separated YYYY-MM-DD list or 'all'", default=None
    )
    qp.add_argument("--horizon-s", type=int, default=30)
    qp.add_argument("--max-files", type=int, default=None)
    qp.add_argument("--logbook-dir", default=None)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    base_dir = args.logbook_dir or resolve_logbook_dir()

    if args.cmd == "logical":
        dates = _parse_dates_arg(args.dates, base_dir, "signal_emitted", args.symbol)
        res = logical_test(
            args.symbol, base_dir=base_dir, dates=dates, max_files=args.max_files
        )
        out = {
            "num_signals": res.num_signals,
            "fields_ok": res.fields_ok,
            "time_monotonic": res.time_monotonic,
            "sample": res.sample.head(10).to_dict(orient="records"),
        }
        print(json.dumps(out, indent=2))
        return 0

    if args.cmd == "quality":
        dates = _parse_dates_arg(args.dates, base_dir, "signal_emitted", args.symbol)
        res = quality_test(
            args.symbol,
            base_dir=base_dir,
            dates=dates,
            horizon_s=args.horizon_s,
            max_files=args.max_files,
        )
        report = res.report
        out = {
            "num_trades": report.num_trades,
            "win_rate": report.win_rate,
            "mean_ret_bps": report.mean_ret_bps,
            "sharpe": report.sharpe,
            "pnl_bps": report.pnl_bps,
            "score": report.score,
        }
        print(json.dumps(out, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
