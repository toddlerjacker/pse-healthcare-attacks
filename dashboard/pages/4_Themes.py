import streamlit as st
import plotly.express as px
import pandas as pd
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import load_data, sidebar_filters

st.set_page_config(page_title="Themes (NLP)", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #f7f5f2; }
    header[data-testid="stHeader"] { background-color: #f7f5f2; }
    [data-testid="stToolbar"] { background-color: #f7f5f2; }
    </style>
""", unsafe_allow_html=True)

# ── Load the labeled topic file ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TOPIC_PATH = BASE_DIR / "data" / "pse_topics_labeled.csv"
# Fallback to flat project dir if not in data/
if not TOPIC_PATH.exists():
    TOPIC_PATH = Path("/Users/andreisales/Desktop/document idf/pse_topics_labeled.csv")

@st.cache_data
def load_topics():
    return pd.read_csv(TOPIC_PATH, parse_dates=["Date"])

df = load_topics()
df = df[df["theme"] != "Unclassified"].copy()

st.title("Discovered Themes: NLP Topic Modeling")
st.markdown(
    "<p style='font-size:13px; color:#888888;'>"
    "Themes surfaced by unsupervised topic modeling (BERTopic) on incident descriptions, "
    "with no keywords supplied. Each incident is assigned to its dominant theme."
    "</p>",
    unsafe_allow_html=True
)

st.divider()

# ── Mode-of-attack metric cards ───────────────────────────────────────────────
mode_counts = df["mode"].value_counts()
total = len(df)

c1, c2, c3 = st.columns(3)
with c1:
    n = int(mode_counts.get("Direct Violence", 0))
    st.metric("Direct Violence", f"{n:,}")
    st.caption(f"{n/total*100:.0f}%: airstrikes, shootings, shelling, residential strikes")
with c2:
    n = int(mode_counts.get("Access Denial", 0))
    st.metric("Access Denial", f"{n:,}")
    st.caption(f"{n/total*100:.0f}%: blocked ambulances, fuel deprivation, checkpoint obstruction")
with c3:
    n = int(mode_counts.get("Detention", 0))
    st.metric("Detention", f"{n:,}")
    st.caption(f"{n/total*100:.0f}%: arrests of health workers")

st.info(
    "**Key finding:** roughly 1 in 5 attacks on healthcare were non-kinetic, "
    "denying access, fuel, or movement rather than striking directly. This systemic "
    "obstruction pattern was surfaced by topic modeling and is not captured by "
    "casualty counts alone."
)

st.divider()

# ── Theme distribution ────────────────────────────────────────────────────────
st.subheader("Incidents by theme")
st.caption("Grouped by mode of attack · themes ordered by size within each mode")

mode_colors = {
    "Direct Violence": "#c0392b",
    "Access Denial":   "#e8916a",
    "Detention":       "#7fb3d3",
    "Other":           "#aaaaaa",
}

dist = (
    df.groupby(["theme", "mode"])
    .size()
    .reset_index(name="Incidents")
)

# Order: group by mode (Detention -> Access Denial -> Direct Violence, bottom-up),
# and within each mode sort by size. Building the y-axis order explicitly puts
# each mode's themes together as a contiguous block.
mode_block_order = ["Direct Violence", "Access Denial", "Detention"]
theme_order = []
for m in reversed(mode_block_order):          # reversed => largest mode ends on top
    block = (
        dist[dist["mode"] == m]
        .sort_values("Incidents", ascending=True)["theme"]
        .tolist()
    )
    theme_order.extend(block)

fig_dist = px.bar(
    dist, x="Incidents", y="theme", orientation="h",
    color="mode", color_discrete_map=mode_colors,
    category_orders={"theme": theme_order, "mode": mode_block_order},
    labels={"theme": "", "mode": ""}, text="Incidents",
)
fig_dist.update_traces(
    textposition="outside", cliponaxis=False,
    hovertemplate="<b>%{y}</b><br>%{x} incidents<extra></extra>"
)
fig_dist.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(gridcolor="#eeeeee"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10, r=80), height=440,
)
st.plotly_chart(fig_dist, width='stretch')

st.divider()

# ── Themes over time (normalized share per month) ────────────────────────────
st.subheader("How themes evolved")
st.caption("Share of each month's incidents by mode of attack · Oct 2023 onward")

df["Year_Month"] = df["Date"].dt.strftime("%Y-%m")

tot = (
    df[df["Year_Month"] >= "2023-10"]
    .groupby(["Year_Month", "mode"])
    .size()
    .reset_index(name="Incidents")
)
tot_total = tot.groupby("Year_Month")["Incidents"].transform("sum")
tot["Share"] = (tot["Incidents"] / tot_total * 100).round(1)

# Force chronological categorical order so the x-axis is discrete, not a 1970+ timeline
month_order = sorted(tot["Year_Month"].unique())
tot["Year_Month"] = pd.Categorical(tot["Year_Month"], categories=month_order, ordered=True)
tot = tot.sort_values("Year_Month")

# Thin x-axis: label roughly every 3rd month so it isn't crowded
tick_vals = month_order[::3]

fig_tot = px.area(
    tot, x="Year_Month", y="Share", color="mode",
    color_discrete_map=mode_colors,
    labels={"Year_Month": "Month", "Share": "% of month", "mode": ""},
    category_orders={"Year_Month": month_order},
)
fig_tot.update_traces(
    hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.1f}%<extra></extra>"
)
fig_tot.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(type="category", tickmode="array", tickvals=tick_vals,
               tickangle=45, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="#eeeeee", ticksuffix="%"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10), height=340,
)
st.plotly_chart(fig_tot, width='stretch')

st.divider()

# ── Theme x Region ────────────────────────────────────────────────────────────
st.subheader("Where each theme occurs")
st.caption("Regional split per theme · Detention and obstruction concentrate in the West Bank; airstrikes in Gaza")

region_df = (
    df.groupby(["theme", "Region"])
    .size()
    .reset_index(name="Incidents")
)
# Keep the two main regions; fold anything else into the order by total size
theme_order = (
    df.groupby("theme").size().sort_values(ascending=True).index.tolist()
)

region_colors = {
    "Gaza Strip": "#c0392b",
    "West Bank":  "#7fb3d3",
}

fig_region = px.bar(
    region_df, x="Incidents", y="theme", color="Region",
    orientation="h", barmode="stack",
    color_discrete_map=region_colors,
    category_orders={"theme": theme_order},
    labels={"theme": "", "Region": ""},
)
fig_region.update_traces(
    hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x}<extra></extra>"
)
fig_region.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(gridcolor="#eeeeee"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10), height=420,
)
st.plotly_chart(fig_region, width='stretch')


st.divider()

# ── Cross-validation: convergence AND divergence, two focused points ──────────
st.subheader("Model validation: does the unsupervised model agree with hand-coding?")

# Compute the two numbers we actually highlight
rc_rate = df.loc[df["theme"] == "Red Cross / Crescent Targeting", "protected_entity"].mean()
obstruction_themes = ["Checkpoint Obstruction", "Ambulance Access Denied", "Fuel & Supply Deprivation"]
flag_cols = ["protected_entity", "residential_strike", "children_affected", "repeat_target_text"]
obstruction_mask = df["theme"].isin(obstruction_themes)
obstruction_count = int(obstruction_mask.sum())
# share of obstruction incidents that NO keyword flag caught
obstruction_missed = int((df.loc[obstruction_mask, flag_cols].sum(axis=1) == 0).sum())

colA, colB = st.columns(2)
with colA:
    st.metric("Convergence", f"{rc_rate*100:.0f}%")
    st.caption(
        "The unsupervised **Red Cross / Crescent** theme overlaps with the independently "
        "hand-coded protected-entity flag. Two methods that share no logic landed on the "
        "same incidents, validating both."
    )
with colB:
    st.metric("Discovery", f"{obstruction_count}")
    st.caption(
        f"Access-denial incidents (checkpoints, ambulances, fuel) the model grouped into a "
        f"coherent theme, of which {obstruction_missed} tripped **none** of the keyword flags. "
        "The model surfaced an entire non-kinetic attack pattern that keyword methods never captured."
    )

st.info(
    "**The two results together are the point:** where the model *should* agree with hand-coding "
    "it agrees strongly (convergence); where keywords were blind, the model found new structure "
    "(discovery). Agreement where expected proves the method is sound; discovery where keywords were blind proves it was worth doing"
)


st.divider()

# ── Theme explorer ────────────────────────────────────────────────────────────
st.subheader("Explore incidents by theme")

theme_pick = st.selectbox(
    "Select a theme",
    sorted(df["theme"].unique()),
)

sub = df[df["theme"] == theme_pick].copy()
sub = sub.sort_values("Health Workers Killed", ascending=False)

st.caption(f"{len(sub):,} incidents in this theme")

for _, row in sub.head(8).iterrows():
    date   = row["Date"].date() if pd.notna(row["Date"]) else "Unknown"
    region = row.get("Region", "Unknown")
    killed = int(row.get("Health Workers Killed", 0) or 0)
    injured = int(row.get("Health Workers Injured", 0) or 0)
    desc   = str(row.get("description_clean", ""))

    cas = []
    if killed:  cas.append(f"<span style='color:#c0392b;font-weight:600'>{killed} killed</span>")
    if injured: cas.append(f"<span style='color:#e8916a;font-weight:600'>{injured} injured</span>")
    cas_str = " · ".join(cas) if cas else "<span style='color:#888'>No casualties recorded</span>"

    st.markdown(
        f"""
        <div style='background:#ffffff;border-left:4px solid #c0392b;
                    border-radius:0 8px 8px 0;padding:12px 16px;margin-bottom:10px;
                    box-shadow:0 1px 3px rgba(0,0,0,0.06);'>
            <div style='display:flex;justify-content:space-between;margin-bottom:5px'>
                <span style='font-weight:600;color:#333;font-size:13px'>{date} · {region}</span>
                <span style='font-size:12px'>{cas_str}</span>
            </div>
            <div style='color:#555;font-size:13px;line-height:1.6'>{desc}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()
st.caption(
    "Method: BERTopic with all-MiniLM-L6-v2 sentence embeddings on incident descriptions. "
    "Themes are unsupervised, discovered from the text, not predefined. Topics were grouped "
    "into three modes of attack for interpretability. Incidents below the model's confidence "
    "threshold (Unclassified) are excluded from this view."
)
