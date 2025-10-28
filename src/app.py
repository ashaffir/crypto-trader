from __future__ import annotations

import asyncio
import os
from typing import Dict

from loguru import logger

from src.config import load_app_config
from src.collector import SpotCollector, FuturesCollector
from src.features import FeatureEngine
from src.logger import ParquetLogbook
from src.control import Control
from src.runtime_config import RuntimeConfigManager
from src.utils.llm_client import LLMClient, LLMConfig
from src.positions import PositionStore
from src.trading import TradingEngine, TraderSettings, load_trader_settings
from src.broker import ExecutionSettings, PaperBroker, BinanceBrokerSkeleton


async def pipeline() -> None:
    cfg = load_app_config()
    out_dir = cfg.storage.get("logbook_dir", "data/logbook")
    os.makedirs(out_dir, exist_ok=True)

    queue: asyncio.Queue = asyncio.Queue(maxsize=10_000)

    # Track current symbols from config; can be hot-reloaded via runtime overrides
    current_symbols: list[str] = list(cfg.symbols)

    # Choose collector based on market mode
    current_market = str(getattr(cfg, "market", "spot")).lower()
    collector = (
        FuturesCollector(current_symbols, vars(cfg.streams), queue)
        if current_market == "futures"
        else SpotCollector(current_symbols, vars(cfg.streams), queue)
    )
    features = FeatureEngine(
        symbols=current_symbols,
        vol_window_s=cfg.features.vol_1s,
        delta_window_s=cfg.features.delta_1s,
        ma_windows=cfg.features.ma,
    )
    logbook = ParquetLogbook(out_dir)
    control = Control()
    runtime_cfg = RuntimeConfigManager()
    # Trading components
    position_store = PositionStore()
    # Execution/broker setup (safe defaults)
    exec_settings = ExecutionSettings.from_overrides(None)
    # keep local current_market in sync
    broker = PaperBroker(position_store, venue=current_market)
    engine = TradingEngine(position_store, TraderSettings(), broker)

    # ---- LLM Recommender State ----
    llm_client: LLMClient | None = None  # single-LLM mode
    llm_active_name: str | None = None
    llm_window_s: int = 30
    llm_refresh_s: int = 5
    # Consensus state
    consensus_enabled: bool = False
    consensus_members: list[str] = []
    consensus_clients: dict[str, LLMClient] = {}
    # Mirror (inverse) mode
    mirror_enabled: bool = False

    def _apply_llm_timing(overrides: Dict | None) -> None:
        nonlocal llm_window_s, llm_refresh_s
        if overrides is None:
            return
        llm = overrides.get("llm") if isinstance(overrides, dict) else None
        if not isinstance(llm, dict):
            return
        try:
            llm_window_s = int(llm.get("window_seconds", llm_window_s) or llm_window_s)
        except Exception:
            pass
        try:
            llm_refresh_s = int(
                llm.get("refresh_seconds", llm_refresh_s) or llm_refresh_s
            )
        except Exception:
            pass

    def _rebuild_llm_client_from_llm_configs(initial: bool = False) -> None:
        nonlocal llm_client, llm_active_name
        try:
            changed, doc = runtime_cfg.load_llm_configs_if_changed()
            if not changed and not initial:
                return
            active = RuntimeConfigManager.get_active_llm_config_from(doc)
            if not isinstance(active, dict):
                return
            name = active.get("name") if isinstance(active.get("name"), str) else None
            conf = LLMConfig(
                base_url=str(active.get("base_url") or ""),
                provider=(active.get("provider") or None),
                api_key=(active.get("api_key") or None),
                model=(active.get("model") or None),
                system_prompt=(active.get("system_prompt") or None),
                user_template=(active.get("user_template") or None),
            )
            if llm_active_name == name and llm_client is not None:
                return
            if llm_client is not None:
                try:
                    import anyio

                    anyio.from_thread.run(llm_client.aclose)
                except Exception:
                    pass
            llm_client = LLMClient(conf)
            # If debug_save_request enabled in runtime_config.json, set path
            try:
                overrides = runtime_cfg.read() or {}
                llm_section = (
                    overrides.get("llm") if isinstance(overrides, dict) else None
                )
                if isinstance(llm_section, dict) and llm_section.get(
                    "debug_save_request"
                ):
                    import os

                    base_dir = runtime_cfg.paths.base_dir
                    os.makedirs(base_dir, exist_ok=True)
                    llm_client.set_debug_save_path(
                        os.path.join(base_dir, "llm_last_request.json")
                    )
                else:
                    llm_client.set_debug_save_path(None)
            except Exception:
                pass
            llm_active_name = name
            logger.info("LLM client ready: from_llm_configs.json")
        except Exception as e:
            logger.warning(f"Failed to load llm_configs.json: {e}")

    def _parse_consensus_overrides() -> tuple[bool, list[str]]:
        try:
            overrides = runtime_cfg.read() or {}
            llm_section = overrides.get("llm") if isinstance(overrides, dict) else None
            cons = (
                llm_section.get("consensus") if isinstance(llm_section, dict) else None
            )
            enabled = (
                bool(cons.get("enabled", False)) if isinstance(cons, dict) else False
            )
            members = cons.get("members") if isinstance(cons, dict) else []
            if not isinstance(members, list):
                members = []
            members = [str(x) for x in members if isinstance(x, (str,)) and x]
            return enabled, members
        except Exception:
            return False, []

    def _rebuild_consensus_clients_if_needed() -> None:
        nonlocal consensus_enabled, consensus_members, consensus_clients
        # Determine current desired consensus settings
        desired_enabled, desired_members = _parse_consensus_overrides()
        # Has llm_configs.json changed?
        try:
            changed_cfg, doc = runtime_cfg.load_llm_configs_if_changed()
        except Exception:
            changed_cfg, doc = (False, runtime_cfg.read_llm_configs())

        # If nothing changed and settings same, keep
        if (
            not changed_cfg
            and desired_enabled == consensus_enabled
            and sorted(desired_members) == sorted(consensus_members)
        ):
            return

        # Update enabled/members
        consensus_enabled = desired_enabled
        consensus_members = desired_members

        # Close old clients
        try:
            import anyio

            for c in consensus_clients.values():
                try:
                    anyio.from_thread.run(c.aclose)
                except Exception:
                    pass
        except Exception:
            pass
        consensus_clients = {}

        if not consensus_enabled or not consensus_members:
            return

        # Build new clients from llm_configs.json for selected members
        try:
            cfgs = (doc or {}).get("configs", []) if isinstance(doc, dict) else []
            by_name: dict[str, dict] = {}
            for cfg in cfgs:
                if isinstance(cfg, dict) and isinstance(cfg.get("name"), str):
                    by_name[cfg["name"]] = cfg
            for name in consensus_members:
                c = by_name.get(name)
                if not isinstance(c, dict):
                    continue
                conf = LLMConfig(
                    base_url=str(c.get("base_url") or ""),
                    provider=(c.get("provider") or None),
                    api_key=(c.get("api_key") or None),
                    model=(c.get("model") or None),
                    system_prompt=(c.get("system_prompt") or None),
                    user_template=(c.get("user_template") or None),
                )
                consensus_clients[name] = LLMClient(conf)
        except Exception:
            consensus_clients = {}

    def _apply_llm_debug_setting() -> None:
        """Enable/disable saving last LLM req/resp based on runtime overrides."""
        nonlocal llm_client, consensus_clients
        try:
            overrides = runtime_cfg.read() or {}
            llm_section = overrides.get("llm") if isinstance(overrides, dict) else None
            dbg_enabled = (
                bool(llm_section.get("debug_save_request"))
                if isinstance(llm_section, dict)
                else False
            )
            import os

            base_dir = runtime_cfg.paths.base_dir
            os.makedirs(base_dir, exist_ok=True)
            if llm_client is not None:
                llm_client.set_debug_save_path(
                    os.path.join(base_dir, "llm_last_request.json")
                    if dbg_enabled
                    else None
                )
            # Apply to consensus clients as well (use per-client filenames to avoid collisions)
            for name, client in consensus_clients.items():
                safe = "llm_last_request" if not name else f"llm_last_request_{name}"
                client.set_debug_save_path(
                    os.path.join(base_dir, f"{safe}.json") if dbg_enabled else None
                )
        except Exception:
            # Never crash the loop on debug setting updates
            pass

    async def llm_loop() -> None:
        nonlocal llm_client, llm_window_s, llm_refresh_s, mirror_enabled
        # Initialize from llm_configs.json before entering loop
        _rebuild_llm_client_from_llm_configs(initial=True)
        _rebuild_consensus_clients_if_needed()
        while True:
            try:
                # Rebuild client(s) if llm_configs.json or consensus changed
                _rebuild_llm_client_from_llm_configs()
                _rebuild_consensus_clients_if_needed()
                # If consensus is enabled and we have members, use consensus mode; else single-LLM mode
                use_consensus = bool(
                    consensus_enabled and consensus_members and consensus_clients
                )
                if not use_consensus and llm_client is None:
                    await asyncio.sleep(1)
                    continue
                # Apply debug save toggle live from runtime_config.json
                _apply_llm_debug_setting()
                # Refresh trader settings from runtime overrides
                try:
                    _changed, _ovr = runtime_cfg.load_if_changed()
                    settings = load_trader_settings(_ovr)
                    engine.update_settings(settings)
                    # Update mirror mode flag from overrides (llm.mirror_mode)
                    try:
                        llm_section = (
                            (_ovr or {}).get("llm") if isinstance(_ovr, dict) else None
                        )
                        mirror_enabled = bool(
                            (llm_section or {}).get("mirror_mode", False)
                        )
                    except Exception:
                        pass
                    # Rebuild broker if execution settings changed
                    try:
                        new_exec = ExecutionSettings.from_overrides(_ovr)
                        nonlocal exec_settings, broker, current_market
                        if new_exec != exec_settings:
                            exec_settings = new_exec
                            if exec_settings.mode == "live":
                                broker = BinanceBrokerSkeleton(
                                    position_store, exec_settings
                                )
                            else:
                                # update venue based on runtime market
                                rt_market = str(
                                    (_ovr or {}).get("market") or current_market
                                ).lower()
                                broker = PaperBroker(position_store, venue=rt_market)
                            engine.broker = broker
                            logger.info(
                                f"Broker switched: mode={exec_settings.mode}, venue={exec_settings.venue}, network={exec_settings.network}"
                            )
                        # Handle market toggle while in paper mode
                        try:
                            rt_market = str(
                                (_ovr or {}).get("market") or current_market
                            ).lower()
                            if (
                                exec_settings.mode != "live"
                                and rt_market != current_market
                            ):
                                current_market = rt_market
                                broker = PaperBroker(
                                    position_store, venue=current_market
                                )
                                engine.broker = broker
                                logger.info(
                                    f"Paper broker venue updated to {current_market}"
                                )
                        except Exception:
                            pass
                    except Exception:
                        pass
                except Exception:
                    pass
                # Aggregate per-symbol summaries and query LLM once per refresh
                summaries = []
                for sym in current_symbols:
                    summary = features.summarize_window(sym, window_s=llm_window_s)
                    if summary and summary.get("count", 0) > 0:
                        summaries.append(summary)
                if summaries:
                    variables = {
                        "symbols": list(current_symbols),
                        "window_seconds": llm_window_s,
                        "DATA_WINDOW": summaries,
                    }
                    now_ms = int(__import__("time").time() * 1000)
                    if use_consensus:
                        # Fan out to all consensus members
                        try:
                            # Run requests concurrently
                            tasks = [
                                c.generate(variables)
                                for c in consensus_clients.values()
                            ]
                            results = await asyncio.gather(
                                *tasks, return_exceptions=True
                            )
                            member_names = list(consensus_clients.keys())
                            # Build per-member per-symbol rec mapping
                            per_member: dict[str, dict[str, dict[str, object]]] = {}
                            for idx, res in enumerate(results):
                                name = member_names[idx]
                                per_member[name] = {}
                                if isinstance(res, Exception) or res is None:
                                    continue
                                for rec in res:
                                    try:
                                        asset = str(rec.get("asset") or "").upper()
                                        if not asset:
                                            continue
                                        direction_raw = str(
                                            rec.get("direction")
                                        ).lower()
                                        direction = (
                                            "buy"
                                            if direction_raw in ("buy", "long")
                                            else "sell"
                                        )
                                        leverage = int(rec.get("leverage") or 1)
                                        conf_val = rec.get("confidence")
                                        conf = None
                                        try:
                                            conf = (
                                                float(conf_val)
                                                if conf_val is not None
                                                else None
                                            )
                                            if conf is not None and (
                                                conf < 0.0 or conf > 1.0
                                            ):
                                                conf = None
                                        except Exception:
                                            conf = None
                                        per_member[name][asset] = {
                                            "direction": direction,
                                            "leverage": leverage,
                                            "confidence": conf,
                                        }
                                    except Exception:
                                        continue
                            # Determine consensus per symbol
                            for sym in current_symbols:
                                sym_upper = str(sym).upper()
                                # Gather member decisions
                                dirs: list[str] = []
                                confs: list[float] = []
                                lev: int | None = None
                                unanimous = True
                                for name in consensus_members:
                                    m = per_member.get(name, {}).get(sym_upper)
                                    if not isinstance(m, dict):
                                        unanimous = False
                                        break
                                    d = str(m.get("direction") or "")
                                    c = m.get("confidence")
                                    if d not in ("buy", "sell"):
                                        unanimous = False
                                        break
                                    if not isinstance(c, (int, float)):
                                        unanimous = False
                                        break
                                    if c < engine.settings.confidence_threshold:
                                        unanimous = False
                                        break
                                    dirs.append(d)
                                    confs.append(float(c))
                                    if lev is None:
                                        try:
                                            lev = int(m.get("leverage") or 1)
                                        except Exception:
                                            lev = 1
                                if not unanimous or not dirs:
                                    # If a consensus-opened position exists, close on first break
                                    # Prefer the specific summary for symbol if available
                                    price_info = None
                                    try:
                                        price_info = next(
                                            (
                                                s
                                                for s in summaries
                                                if s.get("symbol") == sym_upper
                                            ),
                                            None,
                                        )
                                        if (
                                            isinstance(price_info, dict)
                                            and "last_mid" in price_info
                                        ):
                                            price_info = {
                                                "mid": price_info.get("last_mid"),
                                                "last_px": None,
                                            }
                                    except Exception:
                                        price_info = None
                                    try:
                                        engine.maybe_close_on_consensus_break(
                                            symbol=sym_upper,
                                            ts_ms=now_ms,
                                            price_info=price_info,
                                        )
                                    except Exception:
                                        pass
                                    continue
                                # Ensure all directions equal
                                if len(set(dirs)) != 1:
                                    # Consensus broken -> close if needed
                                    price_info = None
                                    try:
                                        price_info = next(
                                            (
                                                s
                                                for s in summaries
                                                if s.get("symbol") == sym_upper
                                            ),
                                            None,
                                        )
                                        if (
                                            isinstance(price_info, dict)
                                            and "last_mid" in price_info
                                        ):
                                            price_info = {
                                                "mid": price_info.get("last_mid"),
                                                "last_px": None,
                                            }
                                    except Exception:
                                        price_info = None
                                    try:
                                        engine.maybe_close_on_consensus_break(
                                            symbol=sym_upper,
                                            ts_ms=now_ms,
                                            price_info=price_info,
                                        )
                                    except Exception:
                                        pass
                                    continue
                                # All good: open decision
                                direction = dirs[0]
                                leverage = int(lev or 1)
                                consensus_conf = min(confs) if confs else None
                                # Log consensus recommendation
                                logbook.append_trade_recommendation(
                                    [
                                        {
                                            "ts_ms": now_ms,
                                            "symbol": sym_upper,
                                            "asset": sym_upper,
                                            "direction": direction,
                                            "leverage": leverage,
                                            "llm_model": "consensus-llm",
                                            **(
                                                {"confidence": consensus_conf}
                                                if consensus_conf is not None
                                                else {}
                                            ),
                                            "source": "llm",
                                        }
                                    ]
                                )
                                # price info for this symbol
                                price_info = None
                                try:
                                    price_info = next(
                                        (
                                            s
                                            for s in summaries
                                            if s.get("symbol") == sym_upper
                                        ),
                                        None,
                                    )
                                    if (
                                        isinstance(price_info, dict)
                                        and "last_mid" in price_info
                                    ):
                                        price_info = {
                                            "mid": price_info.get("last_mid"),
                                            "last_px": None,
                                        }
                                except Exception:
                                    price_info = None
                                # Close inverse or TP/SL (use mirrored recommendation under mirror mode)
                                _close_rec_dir = (
                                    ("sell" if direction == "buy" else "buy")
                                    if mirror_enabled
                                    else direction
                                )
                                engine.maybe_close_on_inverse_or_tp_sl(
                                    symbol=sym_upper,
                                    recommendation_direction=_close_rec_dir,
                                    confidence=consensus_conf,
                                    ts_ms=now_ms,
                                    price_info=price_info,
                                )
                                # Try open (apply mirror mode to execution)
                                _exec_dir = (
                                    ("sell" if direction == "buy" else "buy")
                                    if mirror_enabled
                                    else direction
                                )
                                engine.maybe_open_from_recommendation(
                                    symbol=sym_upper,
                                    direction=_exec_dir,
                                    leverage=leverage,
                                    confidence=consensus_conf,
                                    ts_ms=now_ms,
                                    price_info=price_info,
                                    llm_model="consensus-llm",
                                    llm_window_s=llm_window_s,
                                )
                        except Exception:
                            # Do not break the loop on consensus errors
                            pass
                    else:
                        # Single LLM mode
                        recs = await llm_client.generate(variables)
                        if recs:
                            for rec in recs:
                                try:
                                    asset = str(rec.get("asset") or "").upper()
                                    # Fallback to first summary symbol if asset missing
                                    if not asset and summaries:
                                        asset = str(
                                            summaries[0].get("symbol") or ""
                                        ).upper()
                                    if not asset:
                                        continue
                                    direction_raw = str(rec.get("direction")).lower()
                                    direction = (
                                        "buy"
                                        if direction_raw in ("buy", "long")
                                        else "sell"
                                    )
                                    leverage = int(rec.get("leverage") or 1)
                                    # Optional confidence from LLM (0..1)
                                    conf_val = rec.get("confidence")
                                    try:
                                        confidence = (
                                            float(conf_val)
                                            if conf_val is not None
                                            else None
                                        )
                                        if confidence is not None:
                                            if confidence < 0.0 or confidence > 1.0:
                                                confidence = None
                                    except Exception:
                                        confidence = None
                                    logbook.append_trade_recommendation(
                                        [
                                            {
                                                "ts_ms": now_ms,
                                                "symbol": asset,
                                                "asset": asset,
                                                "direction": direction,
                                                "leverage": leverage,
                                                # Persist model used for this recommendation
                                                **(
                                                    {
                                                        "llm_model": str(
                                                            llm_client.cfg.model
                                                        )
                                                    }
                                                    if getattr(llm_client, "cfg", None)
                                                    and getattr(
                                                        llm_client.cfg, "model", None
                                                    )
                                                    else {}
                                                ),
                                                **(
                                                    {"confidence": confidence}
                                                    if confidence is not None
                                                    else {}
                                                ),
                                                "source": "llm",
                                            }
                                        ]
                                    )

                                    # ---- Decision Flow ----
                                    # 1) Close existing if inverse or TP/SL
                                    price_info = None
                                    try:
                                        # Prefer the specific summary for symbol if available
                                        price_info = next(
                                            (
                                                s
                                                for s in summaries
                                                if s.get("symbol") == asset
                                            ),
                                            None,
                                        )
                                        # Map summary to numeric price snapshot for engine
                                        if isinstance(price_info, dict):
                                            # Legacy path: if last_mid present, use it directly
                                            if "last_mid" in price_info:
                                                price_info = {
                                                    "mid": price_info.get("last_mid"),
                                                    "last_px": None,
                                                }
                                            else:
                                                # New compact stats schema: mid is a dict
                                                mid_val = None
                                                try:
                                                    mid_field = price_info.get("mid")
                                                    if isinstance(mid_field, dict):
                                                        # Prefer mean; fallback to max/min
                                                        mid_val = (
                                                            mid_field.get("mean")
                                                            if mid_field.get("mean")
                                                            is not None
                                                            else (
                                                                mid_field.get("max")
                                                                if mid_field.get("max")
                                                                is not None
                                                                else mid_field.get(
                                                                    "min"
                                                                )
                                                            )
                                                        )
                                                    elif isinstance(
                                                        mid_field, (int, float)
                                                    ):
                                                        mid_val = float(mid_field)
                                                except Exception:
                                                    mid_val = None
                                                if mid_val is not None:
                                                    price_info = {
                                                        "mid": mid_val,
                                                        "last_px": None,
                                                    }
                                    except Exception:
                                        price_info = None

                                    # Close inverse or TP/SL (direction mirrored if mirror mode)
                                    _close_rec_dir = (
                                        ("sell" if direction == "buy" else "buy")
                                        if mirror_enabled
                                        else direction
                                    )
                                    engine.maybe_close_on_inverse_or_tp_sl(
                                        symbol=asset,
                                        recommendation_direction=_close_rec_dir,
                                        confidence=confidence,
                                        ts_ms=now_ms,
                                        price_info=price_info,
                                    )

                                    # 2) If no open or slot available and confidence ok -> open
                                    # Apply mirror mode for execution direction
                                    _exec_dir = (
                                        ("sell" if direction == "buy" else "buy")
                                        if mirror_enabled
                                        else direction
                                    )
                                    engine.maybe_open_from_recommendation(
                                        symbol=asset,
                                        direction=_exec_dir,
                                        leverage=leverage,
                                        confidence=confidence,
                                        ts_ms=now_ms,
                                        price_info=price_info,
                                        llm_model=str(
                                            getattr(llm_client.cfg, "model", "")
                                        ),
                                        llm_window_s=llm_window_s,
                                    )
                                except Exception:
                                    # Ignore malformed recs
                                    pass
            except Exception as e:
                logger.warning(f"LLM loop error: {e}")
            finally:
                await asyncio.sleep(max(1, int(llm_refresh_s)))

    async def consumer() -> None:
        batch_snapshots: list[Dict] = []
        last_flush_ts_ms: int | None = None
        while True:
            msg = await queue.get()
            snap = features.on_message(msg)
            if not snap:
                queue.task_done()
                continue

            # Append to market_snapshot batch
            batch_snapshots.append(snap)

            # Flush periodically based on ts or batch size
            ts_ms = snap.get("ts_ms") or last_flush_ts_ms
            should_flush = False
            if ts_ms is not None and last_flush_ts_ms is not None:
                should_flush = (ts_ms - last_flush_ts_ms) >= 1000
            if last_flush_ts_ms is None:
                last_flush_ts_ms = ts_ms

            if should_flush or len(batch_snapshots) >= 2000:
                try:
                    if batch_snapshots:
                        # Group by symbol to avoid cross-partition contamination
                        snaps_by_sym: Dict[str, list[Dict]] = {}
                        for r in batch_snapshots:
                            s = str(r.get("symbol") or "").upper()
                            if not s:
                                continue
                            snaps_by_sym.setdefault(s, []).append(r)
                        for _sym, rows in snaps_by_sym.items():
                            logbook.append_market_snapshot(rows)
                        batch_snapshots.clear()
                except Exception as e:
                    logger.exception(f"Logbook write failed: {e}")
                finally:
                    last_flush_ts_ms = ts_ms

            queue.task_done()

    # Collector task is managed separately to allow restart when symbols change
    collector_task = asyncio.create_task(collector.run(), name="collector")
    tasks = [
        asyncio.create_task(consumer(), name="consumer"),
        asyncio.create_task(llm_loop(), name="llm"),
    ]

    try:
        # Periodically write a heartbeat while running
        while True:
            hb = {
                "status": "running",
                "queue_size": queue.qsize(),
                "symbols": current_symbols,
            }
            control.write_status(hb)

            # Hot-reload runtime overrides if changed (timings, thresholds, horizons)
            try:
                changed, overrides = runtime_cfg.load_if_changed()
                if changed and overrides:
                    _apply_llm_timing(overrides)

                    # Handle tracked symbols change
                    try:
                        new_syms = overrides.get("symbols")
                        if isinstance(new_syms, list):
                            norm = [str(s).upper() for s in new_syms if s]
                            if norm and norm != current_symbols:
                                # Restart collector and rebuild features atomically
                                current_symbols = norm
                                try:
                                    await collector.stop()
                                except Exception:
                                    pass
                                try:
                                    collector_task.cancel()
                                except Exception:
                                    pass
                                # Recreate collector preserving market mode
                                market = str(getattr(cfg, "market", "spot")).lower()
                                collector = (
                                    FuturesCollector(
                                        current_symbols, vars(cfg.streams), queue
                                    )
                                    if market == "futures"
                                    else SpotCollector(
                                        current_symbols, vars(cfg.streams), queue
                                    )
                                )
                                features = FeatureEngine(
                                    symbols=current_symbols,
                                    vol_window_s=cfg.features.vol_1s,
                                    delta_window_s=cfg.features.delta_1s,
                                    ma_windows=cfg.features.ma,
                                )
                                collector_task = asyncio.create_task(
                                    collector.run(), name="collector"
                                )
                    except Exception:
                        # Never crash hot-reload loop on symbol update
                        pass
            except Exception:
                # Swallow to avoid impacting main loop
                pass
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await collector.stop()
        try:
            collector_task.cancel()
        except Exception:
            pass
        for t in tasks:
            t.cancel()


def main() -> None:
    try:
        asyncio.run(pipeline())
    except KeyboardInterrupt:
        logger.info("Shutting down")


if __name__ == "__main__":
    main()
