"""
feature_engineering.py
PSE Healthcare Attacks Dataset — Data Cleaning & Feature Engineering
Run from: /Users/andreisales/Desktop/document idf/
Output:   pse_healthcare_featured.csv
"""

import pandas as pd
import numpy as np
import re

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(
    "/Users/andreisales/Desktop/document idf/pse_healthcare_master_clean.csv",
    parse_dates=["Date"],
    low_memory=False
)
print(f"  Rows: {len(df):,}  |  Cols: {len(df.columns)}")

# ── 1. TIME FEATURES ─────────────────────────────────────────────────────────
print("\n[1] Time features...")

df["Year"]       = df["Date"].dt.year
df["Month"]      = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%B")
df["Year_Month"] = df["Date"].dt.to_period("M").astype(str)   # "2023-10"
df["Quarter"]    = df["Date"].dt.to_period("Q").astype(str)   # "2023Q4"
df["Week"]       = df["Date"].dt.isocalendar().week.astype(int)
df["Day_of_Week"]= df["Date"].dt.day_name()

# Conflict phase labels
def conflict_phase(d):
    if d < pd.Timestamp("2023-10-07"):  return "Before Oct 7 2023"
    if d < pd.Timestamp("2024-01-01"):  return "Oct 7 Aftermath (Oct–Dec 2023)"
    if d < pd.Timestamp("2024-07-01"):  return "Jan–Jun 2024"
    return "Jul 2024 onward"

df["post_oct7"]      = (df["Date"] >= "2023-10-07").astype(int)
df["Period_Binary"]  = df["post_oct7"].map({0: "Pre Oct 7 2023", 1: "Post Oct 7 2023"})
df["Conflict_Phase"] = df["Date"].apply(conflict_phase)

print(f"  Period_Binary counts:\n{df['Period_Binary'].value_counts().to_string()}")

# ── 2. WEAPON CATEGORY ───────────────────────────────────────────────────────
print("\n[2] Weapon categories...")

WEAPON_MAP = {
    "aerial": "Airstrike",
    "air bomb": "Airstrike",
    "drone": "Airstrike",
    "missile": "Airstrike",
    "warplane": "Airstrike",
    "artillery": "Artillery/Shelling",
    "shell": "Artillery/Shelling",
    "tank": "Artillery/Shelling",
    "mortar": "Artillery/Shelling",
    "cannon": "Artillery/Shelling",
    "gunfire": "Gunfire/Small Arms",
    "firearm": "Gunfire/Small Arms",
    "shooting": "Gunfire/Small Arms",
    "bullet": "Gunfire/Small Arms",
    "sniper": "Gunfire/Small Arms",
    "explosive": "Explosive/IED",
    "ied": "Explosive/IED",
    "grenade": "Explosive/IED",
    "bomb": "Explosive/IED",
    "landmine": "Explosive/IED",
    "knife": "Blunt/Sharp Object",
    "blunt": "Blunt/Sharp Object",
    "baton": "Blunt/Sharp Object",
    "tear gas": "Chemical/Crowd Control",
    "rubber": "Chemical/Crowd Control",
    "chemical": "Chemical/Crowd Control",
    "arson": "Arson/Fire",
    "fire": "Arson/Fire",
    "no direct": "No Direct Violence",
    "not applicable": "No Direct Violence",
    "unknown": "Unknown",
    "rocket":   "Explosive/IED",      # was going to Other
    "rpg":      "Explosive/IED",      # was going to Other
    "mine":     "Explosive/IED",      # was going to Other
    "fist":     "Blunt/Sharp Object", # was going to Other
    "foot":     "Blunt/Sharp Object",
    "stone":    "Blunt/Sharp Object",
    "stick":    "Blunt/Sharp Object",
    "gravel":   "Blunt/Sharp Object",
    "no information": "Unknown",
    "other weapon":   "Unknown",
}

def bucket_weapon(w):
    if pd.isna(w): return "Unknown"
    w_lower = str(w).lower()
    for key, bucket in WEAPON_MAP.items():
        if key in w_lower:
            return bucket
    return "Other"

df["Weapon_Category"] = df["Weapon Carried/Used"].apply(bucket_weapon)
print(f"  Weapon_Category distribution:\n{df['Weapon_Category'].value_counts().to_string()}")

# ── 3. PERPETRATOR SIMPLIFICATION ────────────────────────────────────────────
print("\n[3] Perpetrator simplification...")

def simplify_perp(p):
    if pd.isna(p): return "Unknown"
    p_lower = str(p).lower()
    if any(x in p_lower for x in ["israeli defence", "israeli defense", "idf",
                                    "government of israel"]):
        return "Israeli Forces"
    if "israeli settler" in p_lower:          # was checking "settler" — now exact
        return "Israeli Settlers"
    if "israeli police" in p_lower:           # was checking "police" alone
        return "Israeli Police/Border"
    if any(x in p_lower for x in ["hamas", "pij", "islamic jihad", "al-qassam",
                                    "qassam", "al-aqsa", "non-state armed"]):
        return "Palestinian Armed Groups"
    if any(x in p_lower for x in ["unknown", "unidentified", "not reported",
                                    "no information"]):
        return "Unknown"
    if any(x in p_lower for x in ["palestinian national", "palestinian authority"]):
        return "Palestinian Authority"
    return "Other"
df["Perpetrator_Simple"] = df["Reported Perpetrator Name"].apply(simplify_perp)
print(f"  Perpetrator_Simple distribution:\n{df['Perpetrator_Simple'].value_counts().to_string()}")

# ── 4. CASUALTY & SEVERITY FEATURES ─────────────────────────────────────────
print("\n[4] Casualty features...")

casualty_cols = [
    "Health Workers Killed", "Health Workers Injured",
    "Health Workers Kidnapped", "Health Workers Arrested",
    "Health Workers Threatened", "Health Workers Assaulted",
]
for col in casualty_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

df["Total_Casualties"] = (
    df["Health Workers Killed"]   +
    df["Health Workers Injured"]  +
    df["Health Workers Kidnapped"]+
    df["Health Workers Arrested"]
)

df["Severity_Score"] = (
    df["Health Workers Killed"]    * 4 +
    df["Health Workers Injured"]   * 2 +
    df["Health Workers Kidnapped"] * 2 +
    df["Health Workers Arrested"]  * 1 +
    df["Health Workers Threatened"]* 0.5 +
    df["Health Workers Assaulted"] * 1
)

df["Severity_Label"] = pd.cut(
    df["Severity_Score"],
    bins=[-0.1, 0, 2, 6, 9999],
    labels=["None", "Low", "Medium", "High"]
).astype(str)

df["Has_Casualties"]  = (df["Total_Casualties"] > 0).astype(int)
df["Lethal_Incident"] = (df["Health Workers Killed"] > 0).astype(int)

print(f"  Severity_Label distribution:\n{df['Severity_Label'].value_counts().to_string()}")
print(f"  Total lethal incidents: {df['Lethal_Incident'].sum():,}")

# ── 5. INCIDENT TYPE FLAGS ───────────────────────────────────────────────────
print("\n[5] Incident type flags...")

def to_bool_int(col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

df["Facility_Destroyed"] = to_bool_int("Number of Attacks on Health Facilities Reporting Destruction")
df["Facility_Damaged"]   = to_bool_int("Number of Attacks on Health Facilities Reporting Damaged")
df["Forceful_Entry"]     = to_bool_int("Forceful Entry into Health Facility")
df["Transport_Destroyed"]= to_bool_int("Health Transportation Destroyed")
df["Transport_Damaged"]  = to_bool_int("Health Transportation Damaged")
df["Transport_Stolen"]   = to_bool_int("Health Transportation Stolen/Hijacked")

df["Facility_Attack"] = (
    (df["Facility_Destroyed"] > 0) |
    (df["Facility_Damaged"] > 0)   |
    (df["Forceful_Entry"] > 0)
).astype(int)

df["Transport_Attack"] = (
    (df["Transport_Destroyed"] > 0) |
    (df["Transport_Damaged"] > 0)   |
    (df["Transport_Stolen"] > 0)
).astype(int)

df["Access_Disruption"] = df.get(
    "Affects Access to Medicines, Vaccines And Other Health Products",
    pd.Series(0, index=df.index)
).apply(lambda x: 1 if str(x).lower() in ["true", "1", "yes"] else 0)

# ── 6. REGION FEATURES ───────────────────────────────────────────────────────
print("\n[6] Region features...")

def clean_region(r):
    if pd.isna(r): return "Unknown"
    r = str(r).strip()
    if "gaza" in r.lower(): return "Gaza Strip"
    if "west bank" in r.lower() or "westbank" in r.lower(): return "West Bank"
    return r

df["Region"] = df["Admin 1"].apply(clean_region)
print(f"  Region distribution:\n{df['Region'].value_counts().to_string()}")

# ── 7. TEXT FEATURES ─────────────────────────────────────────────────────────
print("\n[7] Text features...")

desc_col = "description_clean" if "description_clean" in df.columns else "Event Description"

df["Desc_Word_Count"] = df[desc_col].fillna("").apply(lambda x: len(str(x).split()))
df["Desc_Char_Count"] = df[desc_col].fillna("").apply(len)

# Keyword flags from description
keywords = {
    "hospital":    r"\bhospital\b",
    "ambulance":   r"\bambulance\b",
    "doctor":      r"\bdoctor|physician\b",
    "nurse":       r"\bnurse\b",
    "paramedic":   r"\bparamedic|ems\b",
    "red_crescent":r"\bred crescent|icrc|red cross\b",
    "un_staff":    r"\bunrwa|united nations|un staff\b",
    "mass_casualty":r"\bmass casualty|multiple killed|dozens\b",
}

for flag, pattern in keywords.items():
    df[f"kw_{flag}"] = df[desc_col].fillna("").str.contains(
        pattern, case=False, regex=True
    ).astype(int)

# ── 8. MAP COLOR COLUMN ───────────────────────────────────────────────────────
print("\n[8] Map color columns...")

# For Folium dot color
def map_color(row):
    if row["Health Workers Killed"] > 0:  return "red"
    if row["Health Workers Injured"] > 0: return "orange"
    if row["Facility_Attack"] > 0:        return "purple"
    if row["Transport_Attack"] > 0:       return "blue"
    return "#4a90d9"

df["Map_Color"] = df.apply(map_color, axis=1)

# Dot radius scaled by severity
df["Map_Radius"] = df["Severity_Score"].apply(
    lambda s: max(4, min(14, 4 + s * 0.8))
)

# ── 9. ROLLING / CUMULATIVE STATS ────────────────────────────────────────────
print("\n[9] Rolling stats (monthly aggregates saved separately)...")

monthly = (
    df.groupby(["Year_Month", "Region"])
    .agg(
        Incidents      = ("Date", "count"),
        HW_Killed      = ("Health Workers Killed", "sum"),
        HW_Injured     = ("Health Workers Injured", "sum"),
        HW_Arrested    = ("Health Workers Arrested", "sum"),
        Lethal_Count   = ("Lethal_Incident", "sum"),
        Facility_Attacks=("Facility_Attack", "sum"),
        Transport_Attacks=("Transport_Attack","sum"),
    )
    .reset_index()
    .sort_values("Year_Month")
)

monthly.to_csv(
    "/Users/andreisales/Desktop/document idf/pse_monthly_stats.csv",
    index=False
)
print(f"  Monthly stats saved: {len(monthly)} rows")

# ── 10. SAVE FEATURED CSV ─────────────────────────────────────────────────────
print("\n[10] Saving featured dataset...")

# Drop original redundant columns we've replaced
drop_cols = [
    "Number of Attacks on Health Facilities Reporting Destruction",
    "Number of Attacks on Health Facilities Reporting Damaged",
    "Forceful Entry into Health Facility",
    "Health Transportation Destroyed",
    "Health Transportation Damaged",
    "Health Transportation Stolen/Hijacked",
    "Conflict-Related Violence", "Political-Related Violence",
    "COVID-19-Related Violence", "Ebola-Related Violence",
    "Vaccination-Related Violence", "Polio-Related Violence",
    "Ebola-Related Violence",
    "Country", "Country ISO",
]
drop_cols = [c for c in drop_cols if c in df.columns]
df_out = df.drop(columns=drop_cols)

out_path = "/Users/andreisales/Desktop/document idf/pse_healthcare_featured.csv"
df_out.to_csv(out_path, index=False)

print(f"\n Done!")
print(f"   Input rows:  {len(df):,}")
print(f"   Output cols: {len(df_out.columns)}")
print(f"   Saved to:    {out_path}")
print(f"\nNew columns added:")
new_cols = [c for c in df_out.columns if c not in [
    "Date","Event Description","Admin 1","Latitude","Longitude",
    "Geo Precision","Reported Perpetrator","Reported Perpetrator Name",
    "Weapon Carried/Used","Location of Incident","Health Workers Killed",
    "Health Workers Injured","Health Workers Kidnapped","Health Workers Arrested",
    "Health Workers Threatened","Health Workers Assaulted",
    "Health Workers Sexually Assaulted","Known Kidnapping or Arrest Outcome",
    "Attacks on Emergency Medical Services","Children's Health Services Affected",
    "Attacks On Children's Hospitals","Women's Health Services Affected",
    "Hospital Utilities Affected","SiND Event ID","Date Event Entered",
    "Date Event Modified","_source","Looting/Theft/Robbery/Burglary of Health Supplies",
    "Access Denied or Obstructed","Reported Health Worker Profession",
    "description_clean","sgbv_type","severity_score","severity_label",
    "Affects Access to Medicines, Vaccines And Other Health Products",
    "Occupation of Health Facility","Vicinity of Health Facility Affected",
]]
for c in new_cols:
    print(f"   + {c}")