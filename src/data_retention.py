from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone, timedelta
from typing import Iterable, List, Tuple

from src.utils.logbook_utils import resolve_logbook_dir, iter_date_partitions


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                continue
    return total


def humanize_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    val = float(n)
    while val >= 1024 and i < len(units) - 1:
        val /= 1024.0
        i += 1
    return f"{val:.2f} {units[i]}"


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def prune_by_days(base_dir: str, max_days: int, dry_run: bool) -> List[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_days)
    removed: List[str] = []
    for _table, _sym, date_str, ddir in iter_date_partitions(base_dir):
        try:
            dt = parse_date(date_str)
        except Exception:
            continue
        if dt < cutoff:
            if dry_run:
                print(f"DRY-RUN would remove: {ddir}")
            else:
                shutil.rmtree(ddir, ignore_errors=True)
                removed.append(ddir)
                _cleanup_empty_parents(ddir)
    return removed


def prune_by_minutes(base_dir: str, max_minutes: int, dry_run: bool) -> List[str]:
    """Prune partitions older than N minutes based on date partition granularity.

    Since partitions are daily (date=YYYY-MM-DD), we approximate minute-level pruning
    by translating minutes to a cutoff datetime and removing any partitions strictly
    older than the cutoff midnight. This errs on the conservative side within the
    current layout.
    """
    cutoff_dt = datetime.now(timezone.utc) - timedelta(minutes=max_minutes)
    cutoff_date = cutoff_dt.strftime("%Y-%m-%d")
    removed: List[str] = []
    for _table, _sym, date_str, ddir in iter_date_partitions(base_dir):
        try:
            dt = parse_date(date_str)
        except Exception:
            continue
        # remove if partition date is earlier than the cutoff date
        if dt.strftime("%Y-%m-%d") < cutoff_date:
            if dry_run:
                print(f"DRY-RUN would remove: {ddir}")
            else:
                shutil.rmtree(ddir, ignore_errors=True)
                removed.append(ddir)
                _cleanup_empty_parents(ddir)
    return removed


def _cleanup_empty_parents(ddir: str) -> None:
    # Remove empty date/symbol/table directories up to base logbook dir
    path = ddir
    for _ in range(3):
        path = os.path.dirname(path)
        try:
            if os.path.isdir(path) and not os.listdir(path):
                os.rmdir(path)
        except OSError:
            break


def prune_to_size_cap(base_dir: str, size_cap_bytes: int, dry_run: bool) -> List[str]:
    # Collect partitions with sizes and sort oldest first by date dir name
    parts: List[Tuple[str, str, str, str]] = list(iter_date_partitions(base_dir))
    parts.sort(key=lambda x: x[2])  # by date string
    removed: List[str] = []
    # Compute total size across all partitions
    total = 0
    sizes = {}
    for _t, _s, _d, ddir in parts:
        sz = dir_size_bytes(ddir)
        sizes[ddir] = sz
        total += sz
    if total <= size_cap_bytes:
        return removed
    # Remove oldest partitions until under cap
    for _t, _s, _d, ddir in parts:
        if total <= size_cap_bytes:
            break
        if dry_run:
            print(f"DRY-RUN would remove: {ddir} ({humanize_bytes(sizes[ddir])})")
        else:
            shutil.rmtree(ddir, ignore_errors=True)
            removed.append(ddir)
            total -= sizes[ddir]
            _cleanup_empty_parents(ddir)
    return removed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prune logbook parquet partitions")
    p.add_argument("--base", default=resolve_logbook_dir(), help="Logbook base dir")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--max-days", type=int, help="Keep only last N days")
    g.add_argument("--size-cap", type=str, help="Cap total size, e.g. 50GB, 500MB")
    p.add_argument("--dry-run", action="store_true", help="Print actions only")
    return p


def parse_size_cap(s: str) -> int:
    s = s.strip().upper()
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    for u in ("TB", "GB", "MB", "KB", "B"):
        if s.endswith(u):
            num = float(s[: -len(u)].strip())
            return int(num * units[u])
    # default bytes
    return int(float(s))


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base = args.base
    if args.max_days is not None:
        removed = prune_by_days(base, args.max_days, args.dry_run)
        if args.dry_run:
            print(f"Would remove {len(removed)} partitions (if not dry-run)")
        else:
            print(f"Removed {len(removed)} partitions")
        return 0
    if args.size_cap is not None:
        cap = parse_size_cap(args.size_cap)
        removed = prune_to_size_cap(base, cap, args.dry_run)
        if args.dry_run:
            print(f"Would remove {len(removed)} partitions (if not dry-run)")
        else:
            print(f"Removed {len(removed)} partitions")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
