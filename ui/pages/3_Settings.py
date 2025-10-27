import streamlit as st
import sys as _sys
import os as _os

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from ui.lib.common import PAGE_HEADER_TITLE, render_status_badge
from ui.lib.settings_state import (
    load_tracked_symbols,
    save_tracked_symbols,
    load_llm_settings,
    save_llm_settings,
    load_llm_configs,
    save_llm_configs,
    upsert_llm_config,
    delete_llm_config,
    set_active_llm,
    load_window_seconds,
    save_window_seconds,
    load_trader_settings,
    save_trader_settings,
    load_trader_fee_settings,
    save_trader_fee_settings,
    load_market_mode,
    save_market_mode,
    load_execution_settings,
    save_execution_settings,
)
from ui.lib.logbook_utils import read_latest_file
from ui.lib.common import LOGBOOK_DIR
from src.utils.llm_client import LLMClient, LLMConfig
from src.utils.data_window import construct_data_window
import asyncio
import json
import httpx

# Tiny order helper (uses saved execution settings)
from src.tiny_order import (
    place_spot_market_order as _tiny_place_order,
    _load_execution as _tiny_load_exec,
    _validate_ready as _tiny_validate_exec,
)


st.set_page_config(page_title="Settings", layout="wide")
render_status_badge(st)
st.subheader("Settings")
st.markdown("**Tracked Symbols**")
symbols = load_tracked_symbols()
sym_text = st.text_input(
    "Symbols (comma separated)",
    value=", ".join(symbols) if symbols else "BTCUSDT",
    help="Enter spot or futures symbols like BTCUSDT",
)


def _save_symbols() -> None:
    parts = [s.strip().upper() for s in sym_text.split(",") if s.strip()]
    if not parts:
        st.error("Provide at least one symbol")
        return
    ok = save_tracked_symbols(parts)
    if ok:
        st.success("Symbols saved")
    else:
        st.error("Failed to save symbols")


st.button("Save Symbols", on_click=_save_symbols)

st.divider()

# ---- Two-column layout with vertical separator ----
# Left: LLM/Tracked Symbols, Middle: thin separator, Right: Trader Settings
llm_col, _sep_col, trader_col = st.columns([1, 0.03, 1])

with llm_col:
    st.markdown("**LLM Settings**")

    # Window Size setting
    window_seconds = load_window_seconds()
    new_window = st.number_input(
        "Data Window Size (seconds)",
        min_value=1,
        max_value=3600,
        value=window_seconds,
        step=1,
        help="Number of seconds of market data to send to the LLM for analysis",
    )

    if new_window != window_seconds:
        if save_window_seconds(new_window):
            st.success(f"✓ Window size updated to {new_window} seconds")
            st.rerun()
        else:
            st.error("Failed to save window size")

    # ---- LLM selection and debug options (moved under Data Window) ----
    # Load configs
    llm_configs = load_llm_configs()
    config_names = [cfg.get("name", "") for cfg in llm_configs if cfg.get("name")]

    # Find active LLM
    active_llm = next(
        (cfg.get("name") for cfg in llm_configs if cfg.get("is_active")), None
    )
    if active_llm:
        st.caption(f"Current active: **{active_llm}**")

    # Dropdown options: existing configs + "Create New"
    dropdown_options = config_names + ["Create New"]
    dropdown_placeholder = "Select LLM"

    # Initialize session state for selected LLM
    if "selected_llm" not in st.session_state:
        st.session_state.selected_llm = dropdown_placeholder

    # Initialize modal states
    if "show_llm_modal" not in st.session_state:
        st.session_state.show_llm_modal = False
    if "llm_modal_mode" not in st.session_state:
        st.session_state.llm_modal_mode = "create"  # "create" or "edit"
    if "editing_llm_name" not in st.session_state:
        st.session_state.editing_llm_name = None

    def _on_llm_select():
        selection = st.session_state.llm_dropdown_key
        if selection == "Create New":
            st.session_state.llm_modal_mode = "create"
            st.session_state.editing_llm_name = None
            st.session_state.show_llm_modal = True
            st.session_state.selected_llm = dropdown_placeholder
        else:
            st.session_state.selected_llm = selection

    # Dropdown
    selected = st.selectbox(
        "Select LLM",
        options=[dropdown_placeholder] + dropdown_options,
        index=(
            0
            if st.session_state.selected_llm == dropdown_placeholder
            else ([dropdown_placeholder] + dropdown_options).index(
                st.session_state.selected_llm
            )
        ),
        key="llm_dropdown_key",
        on_change=_on_llm_select,
        label_visibility="collapsed",
    )

    # Buttons below dropdown
    buttons_enabled = (
        st.session_state.selected_llm != dropdown_placeholder
        and st.session_state.selected_llm in config_names
    )

    col1, col2, col3, col4 = st.columns([0.8, 1.3, 0.9, 1.1])

    def _edit_llm():
        st.session_state.llm_modal_mode = "edit"
        st.session_state.editing_llm_name = st.session_state.selected_llm
        st.session_state.show_llm_modal = True

    def _set_default():
        if set_active_llm(st.session_state.selected_llm):
            st.success(f"✓ {st.session_state.selected_llm} set as default")
            st.rerun()
        else:
            st.error("Failed to set default")

    def _delete_llm():
        name = st.session_state.selected_llm
        if delete_llm_config(name):
            st.success(f"✓ {name} deleted")
            st.session_state.selected_llm = dropdown_placeholder
            st.rerun()
        else:
            st.error("Failed to delete LLM config")

    async def _run_test():
        name = st.session_state.selected_llm
        configs = load_llm_configs()
        cfg_dict = next((c for c in configs if c.get("name") == name), None)
        if not cfg_dict:
            st.error("Config not found")
            return

        # Build LLMConfig
        llm_cfg = LLMConfig(
            base_url=cfg_dict.get("base_url", ""),
            provider=cfg_dict.get("provider"),
            api_key=cfg_dict.get("api_key"),
            model=cfg_dict.get("model"),
            system_prompt=cfg_dict.get("system_prompt"),
            user_template=cfg_dict.get("user_template"),
        )

        client = LLMClient(llm_cfg)
        # Honor debug flag during UI test
        try:
            cur_settings = load_llm_settings()
            if bool(cur_settings.get("debug_save_request", False)):
                import os

                from ui.lib.common import CONTROL_DIR as _CTRL

                os.makedirs(_CTRL, exist_ok=True)
                client.set_debug_save_path(os.path.join(_CTRL, "llm_last_request.json"))
        except Exception:
            pass
        try:
            # Get real market data from logbook
            symbols = load_tracked_symbols()
            window_secs = load_window_seconds()

            data_window = construct_data_window(
                base_dir=LOGBOOK_DIR,
                symbols=symbols,
                window_seconds=window_secs,
            )

            test_vars = {"DATA_WINDOW": data_window}
            result = await client.generate(test_vars)
            debug = client.last_debug()

            if result is not None:
                if len(result) > 0:
                    st.success(f"✓ Test successful! Got {len(result)} recommendations")
                else:
                    st.success(
                        "✓ Test successful! LLM returned no opportunities (valid response)"
                    )
                with st.expander("📊 Recommendations"):
                    st.json(result)
            else:
                # Show detailed failure info
                failure_reason = debug.get("failure_reason", "Unknown")
                st.error(f"Test failed - {failure_reason}")

                # Show what we got from the API
                if "extracted_text" in debug and debug["extracted_text"]:
                    st.warning("**LLM Response Text:**")
                    st.code(debug["extracted_text"][:2000], language="text")

                # Show validation failures
                if "validation_failures" in debug and debug["validation_failures"]:
                    st.warning("**Validation Errors:**")
                    for err in debug["validation_failures"]:
                        st.write(f"• {err}")

                # Show parsed JSON if available
                if "parsed_json" in debug:
                    st.info("**Parsed JSON:**")
                    st.json(debug["parsed_json"])

            # Always show full debug in expander
            with st.expander("🔍 Full Debug Info"):
                st.json(debug)

            # Show data window that was sent
            with st.expander("📤 Data Sent to LLM"):
                st.json(data_window)
        except Exception as e:
            st.error(f"Test failed: {e}")
        finally:
            await client.aclose()

    def _run_test_sync():
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Avoid nested event loop; run the async test in a background thread
            import threading

            _err = {"e": None}

            def _runner():
                try:
                    asyncio.run(_run_test())
                except Exception as e:  # noqa: BLE001
                    _err["e"] = e

            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join()
            if _err["e"] is not None:
                st.error(f"Test failed: {_err['e']}")
        else:
            asyncio.run(_run_test())

    with col1:
        st.button("Edit", on_click=_edit_llm, disabled=not buttons_enabled)

    with col2:
        st.button("Set Default", on_click=_set_default, disabled=not buttons_enabled)

    with col3:
        st.button("Delete", on_click=_delete_llm, disabled=not buttons_enabled)

    with col4:
        st.button("Run Test", on_click=_run_test_sync, disabled=not buttons_enabled)

    # Debug: Save last request toggle (persists under runtime_config.json > llm)
    st.markdown("---")
    current_llm_settings = load_llm_settings()
    dbg_enabled = bool(current_llm_settings.get("debug_save_request", False))
    new_dbg = st.toggle(
        "Save last LLM request to file",
        value=dbg_enabled,
        help="When enabled, the last LLM request payload and endpoint will be written to data/control/llm_last_request.json",
    )
    if new_dbg != dbg_enabled:
        ok = save_llm_settings({"debug_save_request": bool(new_dbg)})
        if ok:
            st.success("✓ Debug setting saved")
        else:
            st.error("Failed to save debug setting")

    st.markdown("---")

    st.markdown("**Market Mode**")
    cur_market = load_market_mode()
    new_market = st.radio(
        "Select Market",
        options=["spot", "futures"],
        index=(0 if cur_market == "spot" else 1),
        horizontal=True,
        help="Choose which market to collect from and simulate fees for",
    )
    if new_market != cur_market:
        if save_market_mode(new_market):
            # Keep fee schedule and execution venue in sync with Market Mode
            try:
                # Update fee market (new value should override existing)
                fees_cur = load_trader_fee_settings()
                _ = save_trader_fee_settings({**fees_cur, "market": str(new_market)})
            except Exception:
                pass
            try:
                # Update execution venue
                ex_cur = load_execution_settings()
                _ = save_execution_settings({**ex_cur, "venue": str(new_market)})
            except Exception:
                pass
            st.success(
                f"✓ Market set to {new_market}. Collector, fees, and venue synced."
            )
            # Refresh to reflect updated fee market immediately
            st.rerun()
        else:
            st.error("Failed to save market mode")

    st.markdown("---")
    st.markdown("**Execution Mode**")
    exec_cur = load_execution_settings()
    exec_mode = st.selectbox(
        "Mode",
        options=["paper", "live"],
        index=(0 if exec_cur.get("mode") == "paper" else 1),
        help="Paper = no real orders. Live = send orders (requires keys).",
    )
    st.caption(f"Venue: derived from Market Mode → {cur_market}")
    exec_network = st.selectbox(
        "Network",
        options=["testnet", "mainnet"],
        index=(0 if exec_cur.get("network") == "testnet" else 1),
        help="Use testnet for safe testing.",
    )
    api_key = st.text_input("API Key", value=str(exec_cur.get("api_key") or ""))
    api_secret = st.text_input(
        "API Secret", value=str(exec_cur.get("api_secret") or ""), type="password"
    )
    if st.button("Save Execution Settings"):
        ok = save_execution_settings(
            {
                "mode": exec_mode,
                "venue": str(cur_market),  # derived from Market Mode
                "network": exec_network,
                "api_key": api_key.strip() or None,
                "api_secret": api_secret.strip() or None,
            }
        )
        if ok:
            st.success("✓ Execution settings saved")
        else:
            st.error("Failed to save execution settings")

    # ---- Tiny Order Test (Spot) ----
    st.markdown("")
    with st.expander("Tiny Order Test (Spot)"):
        st.caption(
            "Places a MARKET order using the saved Execution Settings. Uses quote amount (e.g., USDT)."
        )
        ex_saved = _tiny_load_exec()
        net_text = str(ex_saved.get("network", "testnet"))
        mode_text = str(ex_saved.get("mode", "paper"))
        st.write(
            f"Saved mode: `{mode_text}` · network: `{net_text}` · venue: `{ex_saved.get('venue', 'spot')}`"
        )

        default_symbol = (load_tracked_symbols() or ["BTCUSDT"])[0]
        to_symbol = st.text_input("Symbol", value=default_symbol, key="tiny_symbol")
        to_side = st.radio(
            "Side", options=["BUY", "SELL"], horizontal=True, key="tiny_side"
        )
        to_amount = st.number_input(
            "Quote Amount",
            min_value=0.0,
            value=5.0,
            step=0.1,
            help="Amount in quote currency (e.g., USDT)",
            key="tiny_amount",
        )

        # Extra confirmation for mainnet
        confirm_label = (
            "I understand this places a REAL order on MAINNET"
            if net_text == "mainnet"
            else "Confirm"
        )
        confirmed = st.checkbox(confirm_label, value=False, key="tiny_confirm")

        async def _place_async() -> None:
            ex = _tiny_load_exec()
            err = _tiny_validate_exec(ex)
            if err:
                st.error(err)
                return
            try:
                res = await _tiny_place_order(
                    symbol=str(to_symbol).strip().upper(),
                    side=str(to_side).upper(),
                    quote_qty=float(to_amount),
                    ex=ex,
                )
                st.success("✓ Order accepted")
                st.json(res)
            except httpx.HTTPStatusError as e:  # Show Binance error body clearly
                try:
                    st.error(f"HTTP {e.response.status_code}")
                    st.json(e.response.json())
                except Exception:
                    st.error(f"HTTP {e.response.status_code}: {e.response.text}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed: {e}")

        def _place_sync() -> None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import threading

                _err = {"e": None}

                def _runner():
                    try:
                        asyncio.run(_place_async())
                    except Exception as e:  # noqa: BLE001
                        _err["e"] = e

                t = threading.Thread(target=_runner, daemon=True)
                t.start()
                t.join()
                if _err["e"] is not None:
                    st.error(f"Failed: {_err['e']}")
            else:
                asyncio.run(_place_async())

        st.button(
            "Place Tiny Order",
            on_click=_place_sync,
            disabled=(not confirmed) or (str(ex_saved.get("venue", "spot")) != "spot"),
            type="primary",
        )


# Render vertical separator in the thin middle column
with _sep_col:
    st.markdown(
        """
<div style="height:100%; border-left:1px solid rgba(255,255,255,0.08); margin:0 8px;">
&nbsp;
</div>
        """,
        unsafe_allow_html=True,
    )

# ---- Trader Settings ----
with trader_col:
    st.markdown("**Trader Settings**")
    cur = load_trader_settings()

    conc = st.number_input(
        "Concurrent positions allowed",
        min_value=0,
        value=int(cur.get("concurrent_positions", 1)),
        step=1,
        help="Maximum number of open positions at the same time",
    )

    conf = st.number_input(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(cur.get("confidence_threshold", 0.8)),
        step=0.01,
        help="Minimum confidence score required to take a trade",
    )

    size = st.number_input(
        "Default Position Size (USD)",
        min_value=0.0,
        value=float(cur.get("default_position_size_usd", 0.0)),
        step=10.0,
        help="Nominal USD size for new positions (informational)",
    )

    lev = st.number_input(
        "Default Leverage (optional)",
        min_value=0,
        max_value=125,
        value=(
            int(cur.get("default_leverage"))
            if cur.get("default_leverage") not in (None, "")
            else 0
        ),
        step=1,
        help="If > 0, overrides leverage from LLM",
    )

    max_lev = st.number_input(
        "Maximum Leverage (0 = unlimited)",
        min_value=0,
        max_value=125,
        value=int(cur.get("max_leverage", 0) or 0),
        step=1,
        help="If > 0, caps leverage used for positions",
    )

    # Stop Loss with trailing toggle
    sl_col1, sl_col2 = st.columns([1, 1])
    with sl_col1:
        sl_percent = st.number_input(
            "Stop Loss %",
            min_value=0.0,
            value=float(cur.get("sl_percent", 0.0)),
            step=0.1,
            help="Stop loss trigger in percent (e.g., 0.5 = -0.5%)",
        )
    with sl_col2:
        trailing = st.toggle(
            "Trailing SL Enabled",
            value=bool(cur.get("trailing_sl_enabled", False)),
        )

    # Take Profit with disable toggle
    tp_col1, tp_col2 = st.columns([1, 1])
    with tp_col1:
        tp_percent = st.number_input(
            "Take Profit %",
            min_value=0.0,
            value=float(cur.get("tp_percent", 0.0)),
            step=0.1,
            help="Take profit trigger in percent (e.g., 1.0 = +1%)",
        )
    with tp_col2:
        tp_disabled = st.toggle(
            "TP Disabled",
            value=bool(cur.get("tp_disabled", False)),
        )

    auto_expire = st.number_input(
        "Auto expire [Minutes] (optional)",
        min_value=0,
        value=(
            int(cur.get("auto_expire_minutes"))
            if cur.get("auto_expire_minutes") not in (None, "")
            else 0
        ),
        step=1,
        help="If > 0, positions auto-close after this many minutes",
    )

    st.markdown("---")
    st.markdown("**Fee Settings**")
    fees = load_trader_fee_settings()
    fee_enabled = st.toggle(
        "Enable Binance fees in PnL",
        value=bool(fees.get("enabled", False)),
    )
    fee_market = st.selectbox(
        "Market",
        options=["spot", "futures"],
        index=(0 if str(fees.get("market", "spot")) == "spot" else 1),
        help="Select fee schedule. Default is spot.",
    )
    fee_liquidity = st.selectbox(
        "Liquidity",
        options=["taker", "maker"],
        index=(0 if str(fees.get("liquidity", "taker")) == "taker" else 1),
        help="Assume taker (market) or maker (limit) fees",
    )
    fee_vip = st.number_input(
        "VIP Tier",
        min_value=0,
        max_value=9,
        value=int(fees.get("vip_tier", 0)),
        step=1,
    )
    fee_bnb = st.toggle(
        "Apply BNB discount (spot only)",
        value=bool(fees.get("bnb_discount", False)),
    )

    if st.button("Save Trader Settings"):
        payload = {
            "concurrent_positions": int(conc),
            "confidence_threshold": float(conf),
            "default_position_size_usd": float(size),
            "default_leverage": int(lev) if int(lev) > 0 else None,
            "max_leverage": int(max_lev) if int(max_lev) > 0 else 0,
            "tp_percent": float(tp_percent),
            "sl_percent": float(sl_percent),
            "trailing_sl_enabled": bool(trailing),
            "tp_disabled": bool(tp_disabled),
            "auto_expire_minutes": int(auto_expire) if int(auto_expire) > 0 else None,
        }
        if save_trader_settings(payload):
            # Save fees separately to avoid clobbering
            fees_ok = save_trader_fee_settings(
                {
                    "enabled": bool(fee_enabled),
                    "market": str(fee_market),
                    "vip_tier": int(fee_vip),
                    "liquidity": str(fee_liquidity),
                    "bnb_discount": bool(fee_bnb),
                }
            )
            if fees_ok:
                st.success("✓ Trader settings saved (including fees)")
            else:
                st.warning("Trader core settings saved, but failed to save fees")
        else:
            st.error("Failed to save trader settings")


# ---- LLM Modal ----
@st.dialog("LLM Configuration", width="large")
def llm_config_modal():
    mode = st.session_state.llm_modal_mode

    # Load existing config if editing
    if mode == "edit" and st.session_state.editing_llm_name:
        configs = load_llm_configs()
        existing = next(
            (c for c in configs if c.get("name") == st.session_state.editing_llm_name),
            None,
        )
    else:
        existing = None

    # Form fields
    name = st.text_input(
        "LLM Name",
        value=existing.get("name", "") if existing else "",
        disabled=(mode == "edit"),
        help="Unique identifier for this LLM configuration",
    )

    provider = st.selectbox(
        "Provider",
        options=["OpenAI-compatible", "Ollama"],
        index=(
            ["OpenAI-compatible", "Ollama"].index(
                existing.get("provider", "OpenAI-compatible")
            )
            if existing and existing.get("provider") in ["OpenAI-compatible", "Ollama"]
            else 0
        ),
        help="Select the provider type. 'Ollama' uses /api/generate.",
    )

    _model_prefill = existing.get("model", "") if existing else ""
    if not _model_prefill and provider == "Ollama":
        _model_prefill = "qwen2.5:7b"
    model = st.text_input(
        "Model",
        value=_model_prefill,
        help="e.g., gpt-4, claude-3-opus, or Ollama tag like qwen2.5:7b",
    )

    api_key = st.text_input(
        "API Key",
        value=existing.get("api_key", "") if existing else "",
        type="password",
        help="Your API key for this provider",
    )

    # Prefill defaults for Ollama
    default_base = existing.get("base_url", "") if existing else ""
    if not default_base and provider == "Ollama":
        default_base = "https://llm.actappon.com"
    base_url = st.text_input(
        "API Endpoint",
        value=default_base,
        help=(
            "For OpenAI-compatible, host of the API; for Ollama, the server host (we will call /api/generate)."
        ),
    )

    system_prompt = st.text_area(
        "System Prompt",
        value=existing.get("system_prompt", "") if existing else "",
        height=150,
        help="System instructions for the LLM",
    )

    user_template = st.text_area(
        "User Prompt Template",
        value=existing.get("user_template", "") if existing else "",
        height=150,
        help="User prompt template (can use {{variables}} for substitution)",
    )

    col_save, col_cancel = st.columns([1, 1])

    with col_save:
        if st.button("Save", use_container_width=True, type="primary"):
            if not name or not name.strip():
                st.error("LLM Name is required")
            else:
                config = {
                    "name": name.strip(),
                    "provider": provider.strip(),
                    "model": model.strip(),
                    "api_key": api_key,
                    "base_url": base_url.strip(),
                    "system_prompt": system_prompt,
                    "user_template": user_template,
                    "is_active": (
                        existing.get("is_active", False) if existing else False
                    ),
                }

                if upsert_llm_config(config):
                    st.success(f"✓ LLM config saved: {name}")
                    st.session_state.show_llm_modal = False
                    st.session_state.selected_llm = name.strip()
                    st.rerun()
                else:
                    st.error("Failed to save LLM config")

    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.session_state.show_llm_modal = False
            st.rerun()


# Show modal if needed
if st.session_state.show_llm_modal:
    llm_config_modal()

st.divider()
