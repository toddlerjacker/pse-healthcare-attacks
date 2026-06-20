"""
topic_labels.py
─────────────────────────────────────────────────────────────────────
Maps raw BERTopic IDs to human-readable theme labels and grouped
categories, then produces a topic-distribution chart.

Run AFTER nlp_topics.py.  Reads pse_topics.csv.

Outputs:
  pse_topics_labeled.csv   incidents + clean theme label + category
  topic_distribution.html  bar chart of theme sizes
─────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import plotly.express as px

BASE = "/Users/andreisales/Desktop/document idf"

df = pd.read_csv(f"{BASE}/pse_topics.csv", parse_dates=["Date"])

# ── Human-readable labels for each topic id ───────────────────────────────────
# Derived from inspecting topic_summary.csv representative words.
topic_labels = {
    -1: "Unclassified",
     0: "Airstrikes & Killings",
     1: "Red Cross / Crescent Targeting",
     2: "Ambulance Access Denied",
     3: "Arrests & Detentions",
     4: "Fuel & Supply Deprivation",
     5: "Checkpoint Obstruction",
     6: "Attacks on Displaced / Shelters",
     7: "Khan Younis Shootings",
     8: "INGO Facilities Attacked",
     9: "Residential / Home Strikes",
    10: "PRCS Facility / Displacement",
}

# ── Higher-level category grouping ────────────────────────────────────────────
# Collapses 11 themes into 3 modes of attack on healthcare.
topic_category = {
    -1: "Other",
     0: "Direct Violence",
     1: "Direct Violence",
     2: "Access Denial",
     3: "Detention",
     4: "Access Denial",
     5: "Access Denial",
     6: "Direct Violence",
     7: "Direct Violence",
     8: "Direct Violence",
     9: "Direct Violence",
    10: "Direct Violence",
}

# Use the outlier-reduced topic if present, else raw topic
topic_col = "topic_reduced" if "topic_reduced" in df.columns else "topic"

df["theme"]    = df[topic_col].map(topic_labels).fillna("Unclassified")
df["mode"]     = df[topic_col].map(topic_category).fillna("Other")

df.to_csv(f"{BASE}/pse_topics_labeled.csv", index=False)
print(f"Saved pse_topics_labeled.csv  ({len(df):,} rows)")

# ── Distribution chart ────────────────────────────────────────────────────────
dist = (
    df[df["theme"] != "Unclassified"]
    .groupby(["theme", "mode"])
    .size()
    .reset_index(name="Incidents")
    .sort_values("Incidents", ascending=True)
)

mode_colors = {
    "Direct Violence": "#c0392b",
    "Access Denial":   "#e8916a",
    "Detention":       "#7fb3d3",
    "Other":           "#aaaaaa",
}

fig = px.bar(
    dist,
    x="Incidents", y="theme", orientation="h",
    color="mode",
    color_discrete_map=mode_colors,
    labels={"theme": "", "mode": ""},
    text="Incidents",
)
fig.update_traces(
    textposition="outside", cliponaxis=False,
    hovertemplate="<b>%{y}</b><br>%{x} incidents<extra></extra>"
)
fig.update_layout(
    plot_bgcolor="#f7f5f2", paper_bgcolor="#f7f5f2", font_color="#333333",
    xaxis=dict(gridcolor="#eeeeee"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=30, b=10, r=80),
    height=420,
    title="Discovered themes across incident descriptions",
)
fig.write_html(f"{BASE}/topic_distribution.html")
print("Saved topic_distribution.html")

# ── Console summary ───────────────────────────────────────────────────────────
print("\nTheme distribution:")
print(dist.sort_values("Incidents", ascending=False).to_string(index=False))

print("\nMode of attack (grouped):")
print(df["mode"].value_counts().to_string())
