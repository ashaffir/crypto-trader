from __future__ import annotations

import argparse
import os
import sys
import glob
from typing import List

import pyarrow.parquet as pq
import pandas as pd

from src.utils.logbook_utils import resolve_logbook_dir


def list_tables(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    # tables are immediate subdirectories
    return sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )


def list_symbols(base_dir: str, table: str) -> List[str]:
    tdir = os.path.join(base_dir, table)
    if not os.path.isdir(tdir):
        return []
    out: List[str] = []
    for name in os.listdir(tdir):
        if name.startswith("symbol=") and os.path.isdir(os.path.join(tdir, name)):
            out.append(name.split("=", 1)[1])
    return sorted(out)


def list_dates(base_dir: str, table: str, symbol: str) -> List[str]:
    sdir = os.path.join(base_dir, table, f"symbol={symbol}")
    if not os.path.isdir(sdir):
        return []
    out: List[str] = []
    for name in os.listdir(sdir):
        if name.startswith("date=") and os.path.isdir(os.path.join(sdir, name)):
            out.append(name.split("=", 1)[1])
    return sorted(out)


def glob_parquet_files(
    base_dir: str, table: str, symbol: str, date: str | None
) -> List[str]:
    base = os.path.join(base_dir, table, f"symbol={symbol}")
    if date:
        pattern = os.path.join(base, f"date={date}", "*.parquet")
    else:
        pattern = os.path.join(base, "date=*", "*.parquet")
    files = sorted(glob.glob(pattern))
    return files


def cmd_list(args: argparse.Namespace) -> int:
    base = resolve_logbook_dir()
    tables = list_tables(base)
    if not tables:
        print(f"No tables found in {base}")
        return 0
    for t in tables:
        syms = list_symbols(base, t)
        sym_str = ", ".join(syms) if syms else "<none>"
        print(f"{t}: {sym_str}")
    return 0


def cmd_partitions(args: argparse.Namespace) -> int:
    base = resolve_logbook_dir()
    syms = list_symbols(base, args.table)
    if args.symbol and args.symbol not in syms:
        print(
            f"Symbol not found for table {args.table}: {args.symbol}", file=sys.stderr
        )
        return 2
    targets = [args.symbol] if args.symbol else syms
    if not targets:
        print("No symbols found")
        return 0
    for sym in targets:
        dates = list_dates(base, args.table, sym)
        print(f"{args.table}/symbol={sym}: {len(dates)} dates")
        if args.verbose:
            for d in dates:
                print(f"  - {d}")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    base = resolve_logbook_dir()
    files = glob_parquet_files(base, args.table, args.symbol, args.date)
    if not files:
        print("No parquet files found", file=sys.stderr)
        return 2
    try:
        schema = pq.read_schema(files[0])
    except Exception as e:
        print(f"Failed to read schema: {e}", file=sys.stderr)
        return 3
    print(schema)
    return 0


def cmd_head_tail(args: argparse.Namespace, tail: bool) -> int:
    base = resolve_logbook_dir()
    files = glob_parquet_files(base, args.table, args.symbol, args.date)
    if not files:
        print("No parquet files found", file=sys.stderr)
        return 2
    # Try fast path merging; fallback to per-file
    n = args.n
    try:
        df = pq.read_table(files).to_pandas()
    except Exception:
        frames = []
        for f in files:
            try:
                frames.append(pq.read_table(f).to_pandas())
            except Exception:
                continue
        if not frames:
            print("No readable parquet files", file=sys.stderr)
            return 2
        df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.sort_values(df.columns[0]) if df.shape[1] else df
    out = df.tail(n) if tail else df.head(n)
    # Print as pretty table to stdout
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect Parquet datasets in data/logbook")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_list = sub.add_parser("list", help="List tables and symbols")
    sp_list.set_defaults(func=cmd_list)

    sp_part = sub.add_parser(
        "partitions", help="List date partitions for a table[/symbol]"
    )
    sp_part.add_argument("table", help="Table name, e.g. market_snapshot")
    sp_part.add_argument("--symbol", help="Symbol, e.g. BTCUSDT")
    sp_part.add_argument("-v", "--verbose", action="store_true", help="Show dates")
    sp_part.set_defaults(func=cmd_partitions)

    sp_schema = sub.add_parser("schema", help="Show Arrow schema of a dataset")
    sp_schema.add_argument("table")
    sp_schema.add_argument("symbol")
    sp_schema.add_argument("--date", help="Restrict to a specific date (YYYY-MM-DD)")
    sp_schema.set_defaults(func=cmd_schema)

    sp_head = sub.add_parser("head", help="Show first N rows from dataset")
    sp_head.add_argument("table")
    sp_head.add_argument("symbol")
    sp_head.add_argument("-n", type=int, default=20)
    sp_head.add_argument("--date", help="Restrict to date (YYYY-MM-DD)")
    sp_head.set_defaults(func=lambda a: cmd_head_tail(a, tail=False))

    sp_tail = sub.add_parser("tail", help="Show last N rows from dataset")
    sp_tail.add_argument("table")
    sp_tail.add_argument("symbol")
    sp_tail.add_argument("-n", type=int, default=20)
    sp_tail.add_argument("--date", help="Restrict to date (YYYY-MM-DD)")
    sp_tail.set_defaults(func=lambda a: cmd_head_tail(a, tail=True))

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
