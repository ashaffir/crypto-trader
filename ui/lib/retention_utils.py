import os
from datetime import datetime, timezone, timedelta
from typing import Iterable, Tuple

try:
    from src.data_retention import (
        iter_date_partitions,
        prune_by_days,
        prune_to_size_cap,
        parse_size_cap,
        humanize_bytes,
        dir_size_bytes,
    )
except Exception:
    # Lightweight fallbacks for UI-only operation
    def iter_date_partitions(base_dir: str):
        if not os.path.isdir(base_dir):
            return
        for table in sorted(os.listdir(base_dir)):
            tdir = os.path.join(base_dir, table)
            if not os.path.isdir(tdir):
                continue
            for sname in os.listdir(tdir):
                if not sname.startswith("symbol="):
                    continue
                sym = sname.split("=", 1)[1]
                sdir = os.path.join(tdir, sname)
                if not os.path.isdir(sdir):
                    continue
                for dname in os.listdir(sdir):
                    if not dname.startswith("date="):
                        continue
                    date = dname.split("=", 1)[1]
                    ddir = os.path.join(sdir, dname)
                    if os.path.isdir(ddir):
                        yield (table, sym, date, ddir)

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

    def parse_size_cap(s: str) -> int:
        s = s.strip().upper()
        units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
        for u in ("TB", "GB", "MB", "KB", "B"):
            if s.endswith(u):
                num = float(s[: -len(u)].strip())
                return int(num * units[u])
        return int(float(s))

    def prune_by_days(base_dir: str, max_days: int, dry: bool):
        return []

    def prune_to_size_cap(base_dir: str, cap: int, dry: bool):
        return []


__all__ = [
    "iter_date_partitions",
    "prune_by_days",
    "prune_to_size_cap",
    "parse_size_cap",
    "humanize_bytes",
    "dir_size_bytes",
]
