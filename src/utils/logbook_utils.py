from __future__ import annotations

import os
from typing import Iterable, Tuple


def resolve_logbook_dir() -> str:
    return os.getenv("LOGBOOK_DIR", os.path.join("data", "logbook"))


def iter_date_partitions(base_dir: str) -> Iterable[Tuple[str, str, str, str]]:
    """Yield (table, symbol, date, abs_path) for each date partition dir.

    Expects layout: base/table/symbol=<sym>/date=<YYYY-MM-DD>
    """
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


__all__ = ["resolve_logbook_dir", "iter_date_partitions"]
