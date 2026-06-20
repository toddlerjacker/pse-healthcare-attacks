import folium
import pandas as pd
import numpy as np
from folium.plugins import HeatMap, MiniMap, Fullscreen
from folium import LayerControl

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("/Users/andreisales/Desktop/document idf/pse_jittered.csv",
                 parse_dates=["Date"])
print(f"Total rows: {len(df)}")
print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
if "targeting_map" in df.columns:
    print(f"Targeted incidents: {(df['targeting_map'] == 'Targeted').sum()}")

# ── Base map ──────────────────────────────────────────────────────────────────
m = folium.Map(location=[31.8, 35.0], zoom_start=8, tiles=None)
folium.TileLayer("CartoDB dark_matter", name="", show=True).add_to(m)

# ── Fullscreen button ─────────────────────────────────────────────────────────
Fullscreen(position="topleft").add_to(m)

# ── MiniMap ───────────────────────────────────────────────────────────────────
MiniMap(
    tile_layer="CartoDB dark_matter",
    position="bottomright",
    toggle_display=True,
    minimized=False,
    width=150,
    height=100,
    zoom_level_offset=-6
).add_to(m)

# ── Title overlay ─────────────────────────────────────────────────────────────
title_html = """
<div style="
    position: fixed;
    top: 10px; left: 50%;
    transform: translateX(-50%);
    z-index: 1000;
    background: rgba(0,0,0,0.75);
    color: white;
    padding: 8px 16px;
    border-radius: 6px;
    font-family: monospace;
    font-size: 13px;
    pointer-events: none;
    white-space: nowrap;
">
    Attacks on Palestinian Health Workers &nbsp;·&nbsp; Oct 2023 – Feb 2026 &nbsp;·&nbsp;
    <span style='color:#e05c5c'>● Killed</span> &nbsp;
    <span style='color:#d4924a'>● Injured</span> &nbsp;
    <span style='color:#6b9ab8'>● No Casualties</span>
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# ── Credit overlay ────────────────────────────────────────────────────────────
credit_html = """
<div style="
    position: fixed;
    bottom: 10px; left: 10px;
    z-index: 1000;
    background: rgba(0,0,0,0.6);
    color: #aaa;
    padding: 5px 10px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 10px;
    pointer-events: none;
">
    Data: Insecurity Insight / SHCC &nbsp;·&nbsp; 4,230 incidents &nbsp;·&nbsp;
    Source data has 25km precision — exact locations unavailable
</div>
"""
m.get_root().html.add_child(folium.Element(credit_html))

# ── Layer control CSS ─────────────────────────────────────────────────────────
layer_control_css = """
<style>
.leaflet-control-layers {
    background: rgba(15, 15, 15, 0.92) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    padding: 6px 4px !important;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
    min-width: 180px;
}
.leaflet-control-layers-overlays {
    font-family: 'SF Mono', 'Fira Code', monospace !important;
    font-size: 11px !important;
    color: #cccccc !important;
}
.leaflet-control-layers-overlays label {
    display: flex !important;
    align-items: center !important;
    padding: 4px 10px !important;
    margin: 1px 0 !important;
    border-radius: 4px !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
    color: #cccccc !important;
}
.leaflet-control-layers-overlays label:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
}
.leaflet-control-layers-overlays input[type=checkbox] {
    margin-right: 8px !important;
    accent-color: #ff6b6b !important;
    width: 13px !important;
    height: 13px !important;
}
.leaflet-control-layers-separator {
    border-top: 1px solid rgba(255,255,255,0.1) !important;
    margin: 6px 0 !important;
}
.leaflet-control-layers-base {
    display: none !important;
}
.leaflet-control-layers::before {
    content: "LAYERS";
    display: block;
    font-family: 'SF Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.12em;
    color: #666;
    padding: 6px 10px 4px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 4px;
}
</style>
"""
m.get_root().html.add_child(folium.Element(layer_control_css))

# ── Helper: build popup ───────────────────────────────────────────────────────
def make_popup(row, color):
    killed   = int(row.get("Health Workers Killed",  0) or 0)
    injured  = int(row.get("Health Workers Injured", 0) or 0)
    arrested = int(row.get("Health Workers Arrested", 0) or 0)
    desc     = str(row.get("description_clean", ""))[:300]
    date     = row["Date"].date() if pd.notna(row["Date"]) else "Unknown"
    weapon   = row.get("Weapon_Category", row.get("Weapon Carried/Used", "Unknown"))
    perp     = row.get("Perpetrator_Simple", row.get("Reported Perpetrator", "Unknown"))
    region   = row.get("Region", row.get("Admin 1", "Unknown"))
    severity = str(row.get("Severity_Label", "—"))
    severity = "—" if severity == "nan" else severity
    phase    = row.get("Conflict_Phase", "—")

    return folium.Popup(f"""
        <div style='font-family:monospace;font-size:11px;max-width:300px;line-height:1.6'>
        <b style='color:{color};font-size:13px'>{date}</b>
        <span style='float:right;color:#aaa'>{region}</span><br>
        <hr style='margin:3px 0;border-color:#444'>
        <b>Perpetrator:</b> {perp}<br>
        <b>Weapon:</b> {weapon}<br>
        <b>Phase:</b> {phase}<br>
        <hr style='margin:3px 0;border-color:#444'>
        <b>Killed:</b> <span style='color:#e05c5c'>{killed}</span>
        &nbsp;·&nbsp;
        <b>Injured:</b> <span style='color:#d4924a'>{injured}</span>
        &nbsp;·&nbsp;
        <b>Arrested:</b> <span style='color:#6b9ab8'>{arrested}</span><br>
        <b>Severity:</b> {severity}<br>
        <hr style='margin:3px 0;border-color:#444'>
        <span style='color:#ccc'>{desc}</span>
        </div>
    """, max_width=320)

# ── LAYER 1: Density Heatmap ──────────────────────────────────────────────────
heat_fg = folium.FeatureGroup(name="Density Heatmap", show=True)
HeatMap(
    df[["lat_j", "lon_j"]].values.tolist(),
    radius=15, blur=20, min_opacity=0.3,
    gradient={0.2: "blue", 0.5: "orange", 0.8: "red", 1.0: "white"}
).add_to(heat_fg)
heat_fg.add_to(m)

# ── LAYER 2: Casualty markers ─────────────────────────────────────────────────
killed_fg  = folium.FeatureGroup(name="Killed",        show=True)
injured_fg = folium.FeatureGroup(name="Injured",       show=True)
other_fg   = folium.FeatureGroup(name="No Casualties", show=True)

for _, row in df.iterrows():
    killed  = int(row.get("Health Workers Killed",  0) or 0)
    injured = int(row.get("Health Workers Injured", 0) or 0)
    radius  = float(row.get("Map_Radius", 5))

    if killed > 0:
        color, fg = "#e05c5c", killed_fg
    elif injured > 0:
        color, fg = "#d4924a", injured_fg
    else:
        color, fg = "#6b9ab8", other_fg

    folium.CircleMarker(
        location=[row["lat_j"], row["lon_j"]],
        radius=radius,
        color=color, fill=True, fill_color=color,
        fill_opacity=0.55, weight=0,
        popup=make_popup(row, color)
    ).add_to(fg)

killed_fg.add_to(m)
injured_fg.add_to(m)
other_fg.add_to(m)

# ── LAYER 3: Weapon category ──────────────────────────────────────────────────
weapon_colors = {
    "Airstrike":          "#ff4444",
    "Artillery/Shelling": "#ff8800",
    "Gunfire/Small Arms": "#ffcc00",
    "Explosive/IED":      "#cc44ff",
    "No Direct Violence": "#44aaff",
    "Blunt/Sharp Object": "#ff6699",
    "Arson/Fire":         "#ff6600",
}

weapon_fgs = {
    w: folium.FeatureGroup(name=w, show=False)
    for w in weapon_colors
}

for _, row in df.iterrows():
    w = row.get("Weapon_Category", "")
    if w not in weapon_fgs:
        continue
    color = weapon_colors[w]
    folium.CircleMarker(
        location=[row["lat_j"], row["lon_j"]],
        radius=4, color=color, fill=True, fill_color=color,
        fill_opacity=0.55, weight=0,
        popup=make_popup(row, color)
    ).add_to(weapon_fgs[w])

for fg in weapon_fgs.values():
    fg.add_to(m)

# ── LAYER 4: Targeted incidents ───────────────────────────────────────────────
if "targeting_map" in df.columns:
    targeted_fg = folium.FeatureGroup(name="⚠ Targeted incidents", show=False)

    for _, row in df.iterrows():
        if row.get("targeting_map") != "Targeted":
            continue
        folium.CircleMarker(
            location=[row["lat_j"], row["lon_j"]],
            radius=5,
            color="#c0392b", fill=True, fill_color="#c0392b",
            fill_opacity=0.85, weight=0,
            popup=make_popup(row, "#c0392b")
        ).add_to(targeted_fg)

    targeted_fg.add_to(m)
else:
    print("Warning: targeting_map column not found — re-run feature2.py and jitter.py")

# ── Layer control ─────────────────────────────────────────────────────────────
LayerControl(collapsed=True).add_to(m)

# ── Save ──────────────────────────────────────────────────────────────────────
out = "/Users/andreisales/Desktop/document idf/incidents_map.html"
m.save(out)
print(f"Saved → {out}")