import os
import time
import glob
import pyarrow.parquet as pq
import pandas as pd
import streamlit as st

LOGBOOK_DIR = os.getenv("LOGBOOK_DIR", "data/logbook")


def tail_parquet_table(table: str, symbol: str, tail_files: int = 20) -> pd.DataFrame:
    base = os.path.join(LOGBOOK_DIR, table, f"symbol={symbol}")
    files = sorted(glob.glob(os.path.join(base, "date=*", "*.parquet")))
    if not files:
        return pd.DataFrame()
    tbl = pq.read_table(files[-tail_files:])
    return tbl.to_pandas()


st.set_page_config(page_title="Binance Signal Bot", layout="wide")

st.title("Binance Spot Signal Bot — Live Monitor")
symbol = st.sidebar.selectbox("Symbol", ["BTCUSDT"])  # extend later
refresh = st.sidebar.number_input(
    "Auto-refresh (s)", min_value=1, max_value=30, value=2, step=1
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Recent Signals")
    sig_df = tail_parquet_table("signal_emitted", symbol)
    if not sig_df.empty:
        sig_df = sig_df.sort_values("ts_ms", ascending=False).head(50)
        st.dataframe(
            sig_df[["ts_ms", "symbol", "side", "expected_bps", "confidence", "rule_id"]]
        )
    else:
        st.info("No signals yet")

with col2:
    st.subheader("Recent Performance (outcomes)")
    out_df = tail_parquet_table("signal_outcome", symbol)
    if not out_df.empty:
        out_df = out_df.sort_values("resolved_ts_ms", ascending=False).head(100)
        st.metric("Hit-rate (last 100)", f"{(out_df['hit'].mean()*100):.1f}%")
        st.metric("Mean return (bps)", f"{out_df['ret_bps'].mean():.2f}")
        st.dataframe(
            out_df[
                [
                    "signal_id",
                    "resolved_ts_ms",
                    "ret_bps",
                    "hit",
                    "max_adverse_bps",
                    "max_favorable_bps",
                ]
            ]
        )
    else:
        st.info("No outcomes yet")

st.caption("Auto-refreshing…")
time.sleep(refresh)
