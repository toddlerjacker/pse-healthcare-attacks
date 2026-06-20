import pandas as pd
import streamlit as st

from pathlib import Path

# Resolve everything relative to this file so it works locally AND on the cloud
DASHBOARD_DIR = Path(__file__).resolve().parent
REPO_ROOT     = DASHBOARD_DIR.parent
DATA_DIR      = REPO_ROOT / "data"

DATA_PATH   = DATA_DIR / "pse_healthcare_featured_v2.csv"
MAP_PATH    = DATA_DIR / "incidents_map.html"
TOPICS_PATH = DATA_DIR / "pse_topics_labeled.csv"
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df

def sidebar_filters(df):
    st.sidebar.title("Filters")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    # ── Quick filters ─────────────────────────────────────────────────────────
    st.sidebar.markdown("**Quick filters**")
    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("From Oct 7", use_container_width=True):
        st.session_state["start_date"] = pd.Timestamp("2023-10-07").date()
    if col_b.button("All dates", use_container_width=True):
        st.session_state["start_date"] = min_date

    default_start = st.session_state.get("start_date", min_date)
    st.sidebar.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date
    )

    regions = ["All"] + sorted(df["Region"].dropna().unique().tolist())
    region  = st.sidebar.selectbox("Region", regions)

    weapons = ["All"] + sorted(df["Weapon_Category"].dropna().unique().tolist())
    weapon  = st.sidebar.selectbox("Weapon type", weapons)

    perps = ["All"] + sorted(df["Perpetrator_Simple"].dropna().unique().tolist())
    perp  = st.sidebar.selectbox("Perpetrator", perps)

    severities = ["All", "High", "Medium", "Low", "None"]
    severity   = st.sidebar.selectbox("Severity", severities)

    # ── Store in session state for cross-filtering ────────────────────────────
    st.session_state["date_range"] = date_range
    st.session_state["region"]     = region
    st.session_state["weapon"]     = weapon
    st.session_state["perp"]       = perp
    st.session_state["severity"]   = severity

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = df.copy()
    if len(date_range) == 2:
        filtered = filtered[
            (filtered["Date"].dt.date >= date_range[0]) &
            (filtered["Date"].dt.date <= date_range[1])
        ]
    if region != "All":
        filtered = filtered[filtered["Region"] == region]
    if weapon != "All":
        filtered = filtered[filtered["Weapon_Category"] == weapon]
    if perp != "All":
        filtered = filtered[filtered["Perpetrator_Simple"] == perp]
    if severity != "All":
        filtered = filtered[filtered["Severity_Label"] == severity]

    st.sidebar.divider()
    st.sidebar.caption(f"{len(filtered):,} of {len(df):,} incidents shown")

    return filtered


def get_cross_filtered(df, exclude=None):
    """Apply all filters except the excluded dimension."""
    date_range = st.session_state.get("date_range")
    region     = st.session_state.get("region",   "All")
    weapon     = st.session_state.get("weapon",   "All")
    perp       = st.session_state.get("perp",     "All")
    severity   = st.session_state.get("severity", "All")

    filtered = df.copy()

    if date_range and len(date_range) == 2:
        filtered = filtered[
            (filtered["Date"].dt.date >= date_range[0]) &
            (filtered["Date"].dt.date <= date_range[1])
        ]
    if region != "All":
        filtered = filtered[filtered["Region"] == region]
    if weapon != "All" and exclude != "weapon":
        filtered = filtered[filtered["Weapon_Category"] == weapon]
    if perp != "All" and exclude != "perp":
        filtered = filtered[filtered["Perpetrator_Simple"] == perp]
    if severity != "All":
        filtered = filtered[filtered["Severity_Label"] == severity]

    return filtered