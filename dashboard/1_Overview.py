import streamlit as st
import plotly.express as px
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))
from utils import load_data, sidebar_filters, get_cross_filtered

st.set_page_config(
    page_title="Palestine Healthcare Attacks",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #f7f5f2; }
    header[data-testid="stHeader"] { background-color: #f7f5f2; }
    [data-testid="stToolbar"] { background-color: #f7f5f2; }
    </style>
""", unsafe_allow_html=True)

df = load_data()
filtered = sidebar_filters(df)

# ── Cross-filtered versions for weapon + perpetrator charts ───────────────────
weapon_df_full = get_cross_filtered(df, exclude="weapon")
perp_df_full   = get_cross_filtered(df, exclude="perp")

# ── Too few rows warning ──────────────────────────────────────────────────────
if len(filtered) < 10:
    st.warning(f"Only {len(filtered)} incidents match your current filters — too few to visualize meaningfully.")
    st.subheader("Matching incidents")
    st.dataframe(
        filtered[[
            "Date", "Region", "Weapon_Category", "Perpetrator_Simple",
            "Health Workers Killed", "Health Workers Injured",
            "Health Workers Arrested", "Severity_Label", "description_clean"
        ]].sort_values("Date", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=400
    )
    st.stop()

st.title("Attacks on Healthcare — Palestine")
st.markdown(
    "<p style='font-size:13px; color:#888888;'>"
    "Data: Insecurity Insight / SHCC  ·  Oct 2023–Feb 2026<br>"
    "Source data has 25km precision; exact incident locations unavailable. "
    "Points distributed within official PSE boundaries for visualization only."
    "</p>",
    unsafe_allow_html=True
)

st.divider()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Incidents",  f"{len(filtered):,}")
c2.metric("HW Killed",        f"{int(filtered['Health Workers Killed'].sum()):,}")
c3.metric("HW Injured",       f"{int(filtered['Health Workers Injured'].sum()):,}")
c4.metric("HW Arrested",      f"{int(filtered['Health Workers Arrested'].sum()):,}")
c5.metric("Lethal Incidents", f"{int(filtered['Lethal_Incident'].sum()):,}")

st.divider()

# ── Monthly timeline ──────────────────────────────────────────────────────────
st.subheader("Monthly incident timeline")

monthly = (
    filtered.groupby("Year_Month")
    .agg(
        Incidents=("Date", "count"),
        Killed=("Health Workers Killed", "sum"),
        Injured=("Health Workers Injured", "sum"),
    )
    .reset_index()
    .sort_values("Year_Month")
)
monthly["Month_Date"] = pd.to_datetime(monthly["Year_Month"] + "-01")

fig_timeline = px.bar(
    monthly,
    x="Month_Date",
    y="Incidents",
    color="Killed",
    color_continuous_scale=[
        [0.0, "#d0d0d0"],
        [0.1, "#f5c97a"],
        [0.4, "#e8916a"],
        [1.0, "#c0392b"],
    ],
    hover_data=["Killed", "Injured"],
    labels={"Month_Date": "Month", "Incidents": "Incidents", "Killed": "HW Killed"},
)
fig_timeline.update_traces(
    hovertemplate="<b>%{x}</b><br>Incidents: %{y}<br>Killed: %{customdata[0]}<br>Injured: %{customdata[1]}<extra></extra>"
)
fig_timeline.add_vline(
    x=pd.Timestamp("2023-10-07").timestamp() * 1000,
    line_dash="dash",
    line_color="rgba(0,0,0,0.3)",
    annotation_text="Oct 7 2023",
    annotation_position="top right"
)
fig_timeline.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(tickangle=45, tickfont=dict(size=10)),
    coloraxis_colorbar=dict(title="Killed"),
    margin=dict(t=20, b=20), height=320
)
st.plotly_chart(fig_timeline, width='stretch')

# ── Killed vs Injured over time ───────────────────────────────────────────────
st.subheader("Killed vs Injured over time")

kv_monthly = (
    filtered.groupby("Year_Month")
    .agg(
        Killed  =("Health Workers Killed",  "sum"),
        Injured =("Health Workers Injured", "sum"),
    )
    .reset_index()
    .sort_values("Year_Month")
)
kv_monthly["Month_Date"] = pd.to_datetime(kv_monthly["Year_Month"] + "-01")

kv_melt = kv_monthly.melt(
    id_vars="Month_Date",
    value_vars=["Killed", "Injured"],
    var_name="Type",
    value_name="Count"
)

fig_kv = px.area(
    kv_melt,
    x="Month_Date",
    y="Count",
    color="Type",
    color_discrete_map={"Killed": "#c0392b", "Injured": "#e8916a"},
    labels={"Month_Date": "Month", "Count": "Health Workers", "Type": ""},
    line_group="Type",
)
fig_kv.update_traces(
    hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>"
)
fig_kv.add_vline(
    x=pd.Timestamp("2023-10-07").timestamp() * 1000,
    line_dash="dash",
    line_color="rgba(0,0,0,0.4)",
    annotation_text="Oct 7 2023",
    annotation_position="top right"
)
fig_kv.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(tickangle=45, tickfont=dict(size=10), gridcolor="#eeeeee"),
    yaxis=dict(gridcolor="#eeeeee"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10), height=320
)
st.plotly_chart(fig_kv, width='stretch')

st.divider()

# ── Weapon + Perpetrator charts ───────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("By weapon type")
    st.caption("Distribution unaffected by weapon filter")

    w_metric = st.radio(
        "Show by", ["Incidents", "Killed", "Injured"],
        horizontal=True, key="weapon_metric"
    )

    weapon_agg = (
        weapon_df_full.groupby("Weapon_Category")
        .agg(
            Incidents=("Date",                   "count"),
            Killed   =("Health Workers Killed",  "sum"),
            Injured  =("Health Workers Injured", "sum"),
        )
        .reset_index()
        .sort_values(w_metric, ascending=True)
    )

    total_w = weapon_agg[w_metric].sum()
    weapon_agg["Pct"]   = (weapon_agg[w_metric] / total_w * 100).round(1).astype(str) + "%"
    weapon_agg["Label"] = weapon_agg[w_metric].astype(str) + " (" + weapon_agg["Pct"] + ")"

    fig_weapon = px.bar(
        weapon_agg,
        x=w_metric, y="Weapon_Category", orientation="h",
        color=w_metric,
        color_continuous_scale=[[0.0,"#e8e8e8"],[0.5,"#e8916a"],[1.0,"#c0392b"]],
        labels={"Weapon_Category": "", "Killed": "HW Killed", "Injured": "HW Injured"},
        text="Label"
    )
    fig_weapon.update_traces(
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>" + w_metric + ": %{x}<br>%{text}<extra></extra>"
    )
    fig_weapon.update_layout(
        plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
        xaxis=dict(gridcolor="#eeeeee"),
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, r=120), height=380
    )
    st.plotly_chart(fig_weapon, width='stretch')

with col2:
    st.subheader("By perpetrator")
    st.caption("Distribution unaffected by perpetrator filter")

    p_metric = st.radio(
        "Show by", ["Incidents", "Killed", "Injured", "Arrested"],
        horizontal=True, key="perp_metric"
    )

    perp_agg = (
        perp_df_full.groupby("Perpetrator_Simple")
        .agg(
            Incidents=("Date",                    "count"),
            Killed   =("Health Workers Killed",   "sum"),
            Injured  =("Health Workers Injured",  "sum"),
            Arrested =("Health Workers Arrested", "sum"),
        )
        .reset_index()
        .sort_values(p_metric, ascending=True)
    )

    total_p = perp_agg[p_metric].sum()
    perp_agg["Pct"]   = (perp_agg[p_metric] / total_p * 100).round(1).astype(str) + "%"
    perp_agg["Label"] = perp_agg[p_metric].astype(str) + " (" + perp_agg["Pct"] + ")"

    fig_perp = px.bar(
        perp_agg,
        x=p_metric, y="Perpetrator_Simple", orientation="h",
        color=p_metric,
        color_continuous_scale=[[0.0,"#e8e8e8"],[0.5,"#e8916a"],[1.0,"#c0392b"]],
        labels={"Perpetrator_Simple": "", "Killed": "HW Killed",
                "Injured": "HW Injured", "Arrested": "HW Arrested"},
        text="Label"
    )
    fig_perp.update_traces(
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>" + p_metric + ": %{x}<br>%{text}<extra></extra>"
    )
    fig_perp.update_layout(
        plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
        xaxis=dict(gridcolor="#eeeeee"),
        coloraxis_showscale=False,
        margin=dict(t=10, b=10, r=120), height=380
    )
    st.plotly_chart(fig_perp, width='stretch')

st.divider()

# ── Casualties by conflict phase ──────────────────────────────────────────────
st.subheader("Casualties by conflict phase")

phase_order = [
    "Before Oct 7 2023",
    "Oct 7 Aftermath (Oct–Dec 2023)",
    "Jan–Jun 2024",
    "Jul 2024 onward"
]

phase_df = (
    filtered.groupby("Conflict_Phase")
    .agg(
        Incidents=("Date", "count"),
        Killed   =("Health Workers Killed",  "sum"),
        Injured  =("Health Workers Injured", "sum"),
        Arrested =("Health Workers Arrested","sum"),
    )
    .reset_index()
)
phase_df["Conflict_Phase"] = pd.Categorical(
    phase_df["Conflict_Phase"], categories=phase_order, ordered=True
)
phase_df = phase_df.sort_values("Conflict_Phase")

phase_melt = phase_df.melt(
    id_vars="Conflict_Phase",
    value_vars=["Killed", "Injured", "Arrested"],
    var_name="Casualty Type",
    value_name="Count"
)

fig_phase = px.bar(
    phase_melt,
    x="Conflict_Phase", y="Count",
    color="Casualty Type", barmode="group",
    color_discrete_map={
        "Killed":   "#c0392b",
        "Injured":  "#e8916a",
        "Arrested": "#7fb3d3",
    },
    labels={"Conflict_Phase": "", "Count": "Health Workers", "Casualty Type": ""},
)
fig_phase.update_traces(
    hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>"
)
fig_phase.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(tickangle=15, tickfont=dict(size=10)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10), height=320
)
st.plotly_chart(fig_phase, width='stretch')

st.divider()

# ── What was targeted ─────────────────────────────────────────────────────────
st.subheader("What was targeted")

target_data = {
    "Type": [
        "Health Workers Killed",
        "Health Workers Injured",
        "Health Workers Arrested",
        "Facility Attacked",
        "Transport Attacked",
        "Access Disrupted"
    ],
    "Count": [
        int(filtered["Health Workers Killed"].sum()),
        int(filtered["Health Workers Injured"].sum()),
        int(filtered["Health Workers Arrested"].sum()),
        int(filtered["Facility_Attack"].sum()),
        int(filtered["Transport_Attack"].sum()),
        int(filtered["Access_Disruption"].sum()),
    ],
    "Category": [
        "Personnel", "Personnel", "Personnel",
        "Infrastructure", "Infrastructure", "Infrastructure"
    ]
}

target_df = pd.DataFrame(target_data).sort_values("Count", ascending=True)

fig_target = px.bar(
    target_df,
    x="Count", y="Type", orientation="h",
    color="Category",
    color_discrete_map={
        "Personnel":      "#c0392b",
        "Infrastructure": "#7fb3d3",
    },
    labels={"Type": "", "Count": "Count", "Category": ""},
    text="Count"
)
fig_target.update_traces(
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
)
fig_target.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(gridcolor="#eeeeee"),
    yaxis=dict(gridcolor="#eeeeee"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10), height=300
)
st.plotly_chart(fig_target, width='stretch')

st.divider()

# ── Weapon × Conflict Phase heatmap (raw counts) ──────────────────────────────
st.subheader("Weapon use by conflict phase")
st.caption("Cell color = number of incidents · Shows how attack methods shifted across phases")

heat_df = (
    filtered.groupby(["Conflict_Phase", "Weapon_Category"])
    .size()
    .reset_index(name="Incidents")
)
heat_df = heat_df[~heat_df["Weapon_Category"].isin(["Unknown", "Other"])]

pivot = heat_df.pivot_table(
    index="Weapon_Category",
    columns="Conflict_Phase",
    values="Incidents",
    fill_value=0
)
cols = [p for p in phase_order if p in pivot.columns]
pivot = pivot[cols]
pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

fig_heat = px.imshow(
    pivot,
    labels=dict(x="Conflict Phase", y="Weapon", color="Incidents"),
    color_continuous_scale=[
        [0.0,  "#1a1a2e"],
        [0.15, "#2d1b4e"],
        [0.4,  "#8b2635"],
        [0.7,  "#c0392b"],
        [1.0,  "#ff6b6b"],
    ],
    aspect="auto",
    text_auto=True,
)
fig_heat.update_traces(
    textfont=dict(size=11, color="white"),
    hovertemplate="<b>%{y}</b><br>%{x}<br>Incidents: %{z}<extra></extra>"
)
fig_heat.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(tickangle=15, tickfont=dict(size=10), side="bottom"),
    yaxis=dict(tickfont=dict(size=10)),
    coloraxis_colorbar=dict(title="Incidents"),
    margin=dict(t=20, b=20, l=20, r=20),
    height=320
)
st.plotly_chart(fig_heat, width='stretch')

st.divider()

# ── Helper: build monthly weapon pivot filtered to Oct 2023+ ──────────────────
def build_monthly_weapon_pivot(df_in, normalize_rows=False):
    h = (
        df_in.groupby(["Year_Month", "Weapon_Category"])
        .size()
        .reset_index(name="Incidents")
    )
    h = h[~h["Weapon_Category"].isin(["Unknown", "Other"])]
    h = h[h["Year_Month"] >= "2023-10"]

    piv = h.pivot_table(
        index="Weapon_Category",
        columns="Year_Month",
        values="Incidents",
        fill_value=0
    )

    if normalize_rows:
        # Row-wise: % of each weapon's own total per month
        piv_pct = piv.div(piv.sum(axis=1), axis=0) * 100
    else:
        # Column-wise: % of each month's total per weapon
        piv_pct = piv.div(piv.sum(axis=0), axis=1) * 100

    piv_pct = piv_pct.round(1)
    piv_pct = piv_pct.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    return piv_pct

oct7_x = "2023-10"

# ── Heatmap A: column-normalized (% of month) + Oct 7 line ───────────────────
st.subheader("Weapon mix by month — % of monthly incidents")
st.caption("Each cell = share of that month's incidents using this weapon · Oct 7 marker shows escalation point")

pivot_col = build_monthly_weapon_pivot(filtered, normalize_rows=False)

fig_hm_col = px.imshow(
    pivot_col,
    labels=dict(x="Month", y="Weapon", color="% of month"),
    color_continuous_scale=[
        [0.0,  "#1a1a2e"],
        [0.2,  "#2d1b4e"],
        [0.5,  "#8b2635"],
        [0.75, "#c0392b"],
        [1.0,  "#ff6b6b"],
    ],
    aspect="auto",
    text_auto=False,
)
fig_hm_col.update_traces(
    hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.1f}% of that month's incidents<extra></extra>"
)
cols_list_col = list(pivot_col.columns)
if "2023-10" in cols_list_col:
    oct7_idx_col = cols_list_col.index("2023-10")
    fig_hm_col.add_vline(
        x=oct7_idx_col - 0.5,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        annotation_text="Oct 7",
        annotation_font_color="white",
        annotation_position="top left"
    )
fig_hm_col.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(tickangle=45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=10)),
    coloraxis_colorbar=dict(title="% of month"),
    margin=dict(t=30, b=60, l=20, r=20),
    height=320
)
st.plotly_chart(fig_hm_col, width='stretch')

st.divider()

# ── Heatmap B: row-normalized (% of weapon's own total) + Oct 7 line ─────────
st.subheader("Weapon activity by month — % of each weapon's total")
st.caption("Each cell = share of that weapon's all-time incidents in this month · Shows when each weapon peaked relative to itself")

pivot_row = build_monthly_weapon_pivot(filtered, normalize_rows=True)

fig_hm_row = px.imshow(
    pivot_row,
    labels=dict(x="Month", y="Weapon", color="% of weapon total"),
    color_continuous_scale=[
        [0.0,  "#1a1a2e"],
        [0.2,  "#2d1b4e"],
        [0.5,  "#8b2635"],
        [0.75, "#c0392b"],
        [1.0,  "#ff6b6b"],
    ],
    aspect="auto",
    text_auto=False,
)
fig_hm_row.update_traces(
    hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.1f}% of this weapon's total incidents<extra></extra>"
)
cols_list_row = list(pivot_row.columns)
if "2023-10" in cols_list_row:
    oct7_idx_row = cols_list_row.index("2023-10")
    fig_hm_row.add_vline(
        x=oct7_idx_row - 0.5,
        line_dash="dash",
        line_color="rgba(255,255,255,0.5)",
        annotation_text="Oct 7",
        annotation_font_color="white",
        annotation_position="top left"
    )
fig_hm_row.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(tickangle=45, tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=10)),
    coloraxis_colorbar=dict(title="% of weapon total"),
    margin=dict(t=30, b=60, l=20, r=20),
    height=320
)
st.plotly_chart(fig_hm_row, width='stretch')

st.divider()

# ── 100% Stacked Bar — Weapon Lethality ──────────────────────────────────────
st.subheader("Weapon Lethality Profiles")
st.caption("Proportion of killed vs injured per weapon · Wider red = more lethal")

stacked_df = (
    filtered.groupby("Weapon_Category")
    .agg(
        Killed  =("Health Workers Killed",  "sum"),
        Injured =("Health Workers Injured", "sum"),
    )
    .reset_index()
)
stacked_df = stacked_df[
    (stacked_df["Killed"] > 0) | (stacked_df["Injured"] > 0)
]
stacked_df = stacked_df[~stacked_df["Weapon_Category"].isin(["Unknown", "Other"])]
stacked_df["Total"]       = stacked_df["Killed"] + stacked_df["Injured"]
stacked_df["Pct_Killed"]  = (stacked_df["Killed"]  / stacked_df["Total"] * 100).round(1)
stacked_df["Pct_Injured"] = (stacked_df["Injured"] / stacked_df["Total"] * 100).round(1)
stacked_df = stacked_df.sort_values("Pct_Killed", ascending=True)

stacked_melt = stacked_df.melt(
    id_vars=["Weapon_Category", "Total", "Killed", "Injured"],
    value_vars=["Pct_Killed", "Pct_Injured"],
    var_name="Type",
    value_name="Percentage"
)
stacked_melt["Type"] = stacked_melt["Type"].map({
    "Pct_Killed":  "Killed",
    "Pct_Injured": "Injured"
})
stacked_melt["Raw_Count"] = stacked_melt.apply(
    lambda r: r["Killed"] if r["Type"] == "Killed" else r["Injured"],
    axis=1
)

fig_stacked = px.bar(
    stacked_melt,
    x="Percentage",
    y="Weapon_Category",
    color="Type",
    orientation="h",
    barmode="stack",
    color_discrete_map={"Killed": "#c0392b", "Injured": "#e8916a"},
    labels={"Weapon_Category": "", "Percentage": "% of casualties", "Type": ""},
    text="Percentage",
    hover_data=["Raw_Count", "Total"]
)

for _, row in stacked_df.iterrows():
    fig_stacked.add_annotation(
        x=100.5,
        y=row["Weapon_Category"],
        text=f"n={row['Total']}",
        showarrow=False,
        font=dict(size=10, color="#888888"),
        xanchor="left"
    )

fig_stacked.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="inside",
    hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x:.1f}%<br>"
                  "Count: %{customdata[0]}<extra></extra>"
)
fig_stacked.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(gridcolor="#eeeeee", ticksuffix="%", range=[0, 115]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10, r=80),
    height=350
)
st.plotly_chart(fig_stacked, width='stretch')

st.divider()

st.caption(
    "Methodology note: 'Health Workers Killed' reflects the dataset's strict WHO definition of health worker, "
    "excluding civil defense and UN staff. Total casualties in individual incidents may exceed recorded counts. "
    "Over-representation ratio: health workers are ~0.5% of Gaza population but represent 1.69% of deaths (3.4× expected). "
    "Dispersion index = 4.25 (D=1 indicates random; D>1 indicates clustering consistent with deliberate targeting). "
    "Data note: Palestinian Ministry of Health reports 1,581 health workers killed as of July 2025 under a broader "
    "definition. This dataset (770 killed) represents a conservative lower bound using strict WHO criteria."
)
