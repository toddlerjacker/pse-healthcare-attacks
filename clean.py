"""
PSE / Gaza Conflict Datasets — Data Cleaning Pipeline
Run: python clean_datasets.py
Outputs: cleaned CSVs + one merged master file in ./cleaned/

Datasets:
  food      — food systems incidents (Oct 2023 – Sep 2025)
  health    — SHCC health care data 2023–2024 (HXL headers)
  atk26     — attacks on health care 2016–2026 (primary, no text/coords)
  atk25     — attacks on health care 2016–2025 (has 2 extra columns)
  insecurity — InsecurityInsight Oct 2023–Dec 2025 (has text + real coords)
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

BASE = "/Users/andreisales/Desktop/document idf/"
OUT  = os.path.join(BASE, "cleaned")
os.makedirs(OUT, exist_ok=True)

SEP = "=" * 60

def log(msg): print(f"\n{msg}")
def ok(msg):  print(f"  ✓ {msg}")
def warn(msg):print(f"  ⚠ {msg}")

# ── LOAD ──────────────────────────────────────────────────────────────
log(f"{SEP}\nLOADING FILES\n{SEP}")

food   = pd.read_excel(BASE + "2023-2025-pse-gaza-conflict-incidents-affecting-food-systems-incident-data-incident-data.xlsx")
health = pd.read_excel(BASE + "2023-2024-pse-shcc-health-care-data.xlsx")

# health_2324 has a HXL tag row as row 0 — skip it
first_row = health.iloc[0].astype(str).str.contains(r"^#", na=False)
if first_row.sum() > 3:
    health = pd.read_excel(BASE + "2023-2024-pse-shcc-health-care-data.xlsx", header=1)

atk26      = pd.read_excel(BASE + "2016-2026-pse-attacks-on-health-care-incident-data.xlsx")
atk25      = pd.read_excel(BASE + "2016-2025-pse-attacks-on-health-care-incident-data.xlsx")
insecurity = pd.read_excel(BASE + "InsecurityInsight_AttacksonHealthcareOPT (shared on 06.03.26).xlsx")

print(f"  food:       {food.shape}")
print(f"  health:     {health.shape}")
print(f"  atk26:      {atk26.shape}")
print(f"  atk25:      {atk25.shape}")
print(f"  insecurity: {insecurity.shape}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Drop 100% null columns from attacks files
# ══════════════════════════════════════════════════════════════════════
log(f"{SEP}\nSTEP 1 — Drop 100% null columns\n{SEP}")

DROP_COLS = ["Event Description", "Latitude", "Longitude"]

for name, df in [("atk26", atk26), ("atk25", atk25)]:
    dropped = [c for c in DROP_COLS if c in df.columns and df[c].isna().all()]
    for col in dropped:
        df.drop(columns=col, inplace=True)
    ok(f"{name}: dropped {dropped}")

# insecurity has real Event Description + coords — only drop if truly all null
for col in DROP_COLS:
    if col in insecurity.columns and insecurity[col].isna().all():
        insecurity.drop(columns=col, inplace=True)
        ok(f"insecurity: dropped '{col}' (100% null)")
    elif col in insecurity.columns:
        ok(f"insecurity: kept '{col}' ({insecurity[col].notna().sum():,} valid values)")

ok(f"atk26 now {atk26.shape[1]} cols  |  atk25 now {atk25.shape[1]} cols  |  insecurity now {insecurity.shape[1]} cols")


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Rename HXL columns in health_2324
# ══════════════════════════════════════════════════════════════════════
log(f"{SEP}\nSTEP 2 — Rename HXL columns in health_2324\n{SEP}")

HXL_MAP = {
    "#date":                                        "Date",
    "#country +name":                               "Country",
    "#country +ISO":                                "Country ISO",
    "#lat":                                         "Latitude",
    "#long":                                        "Longitude",
    "Unnamed: 5":                                   "Geo Precision",
    "#group +perp ":                                "Reported Perpetrator",
    "#group +perp +name":                           "Reported Perpetrator Name",
    "#weapon":                                      "Weapon Carried/Used",
    "#indicator +health_facility +destroyed +num":  "Number of Attacks on Health Facilities Reporting Destruction",
    "#indicator +health_facility +damaged +num":    "Number of Attacks on Health Facilities Reporting Damaged",
    "#indicator +health_facility +armed_entry +num":"Forceful Entry into Health Facility",
    "#indicator +health_facility +vicinity":        "Vicinity of Health Facility Affected",
    "#indicator +health_facility +occupy":          "Occupation of Health Facility",
    "#indicator +health_transport +destroyed +num": "Health Transportation Destroyed",
    "#indicator +health_transport +damaged +num":   "Health Transportation Damaged",
    "#indicator +health_transport +abducted +num":  "Health Transportation Abducted",
    "#indicator +health_supplies +taken +num":      "Health Supplies Taken",
    "#indicator +health_obstruction":               "Access Denied or Obstructed",
    "#affected +healthworker +killed":              "Health Workers Killed",
    "#affected +healthworker +injured":             "Health Workers Injured",
    "#affected +healthworker +arrested":            "Health Workers Arrested",
    "#affected +healthworker +kidnapped":           "Health Workers Kidnapped",
    "#affected +healthworker +assaulted":           "Known Kidnapping or Arrest Outcome",
    "#affected +healthworker +threatened":          "Health Workers Threatened",
    "#affected +healthworker +assaulted.1":         "Health Workers Assaulted",
    "#affected +healthworker +SGBV":                "Health Workers Sexually Assaulted",
    "#affected +healthworker +profession":          "Health Worker Profession",
    "#event +id":                                   "SiND Event ID",
}

before = health.columns.tolist()
health.rename(columns=HXL_MAP, inplace=True)

renamed = [(b, HXL_MAP[b]) for b in before if b in HXL_MAP]
ok(f"Renamed {len(renamed)} HXL columns")
for old, new in renamed[:6]:
    print(f"    '{old}'  →  '{new}'")
print(f"    ... and {len(renamed)-6} more")


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Standardize Reported Perpetrator labels
# ══════════════════════════════════════════════════════════════════════
log(f"{SEP}\nSTEP 3 — Standardize Reported Perpetrator labels\n{SEP}")

PERP_MAP = {
    # Military
    "Host Government: Military":                         "Government: Military",
    "Host Government/Self-Declared Host Government":     "Government/Self-Declared Government",
    # Add more mappings here if needed
}

for name, df in [("food", food), ("health", health), ("atk26", atk26), ("atk25", atk25), ("insecurity", insecurity)]:
    col = "Reported Perpetrator"
    if col not in df.columns:
        warn(f"{name}: no '{col}' column — skipping")
        continue
    before_vals = df[col].value_counts().to_dict()
    df[col].replace(PERP_MAP, inplace=True)
    after_vals  = df[col].value_counts().to_dict()
    changed = {k: v for k, v in before_vals.items() if k in PERP_MAP}
    if changed:
        ok(f"{name}: standardized {changed}")
    else:
        ok(f"{name}: already clean — no changes needed")


# ══════════════════════════════════════════════════════════════════════
# STEP 3b — Standardize Admin 1 region labels in atk25 + insecurity
# ══════════════════════════════════════════════════════════════════════
log(f"{SEP}\nSTEP 3b — Standardize Admin 1 region labels\n{SEP}")

# atk25 has German-language variants from a different data export
ADMIN_MAP = {
    "Gazastreifen":       "Gaza Strip",
    "Judäa und Samaria":  "Judea and Samaria",
}

for name, df in [("atk25", atk25), ("insecurity", insecurity)]:
    if "Admin 1" not in df.columns:
        warn(f"{name}: no 'Admin 1' column — skipping")
        continue
    before = df["Admin 1"].value_counts().to_dict()
    df["Admin 1"].replace(ADMIN_MAP, inplace=True)
    changed = {k: v for k, v in before.items() if k in ADMIN_MAP}
    if changed:
        ok(f"{name}: standardized {changed}")
    else:
        ok(f"{name}: already clean — no changes needed")

# ══════════════════════════════════════════════════════════════════════
log(f"{SEP}\nSTEP 4 — Merge unique atk25 columns into atk26\n{SEP}")

UNIQUE_25_COLS = ["SiND Event ID", "Access Denied or Obstructed", "Looting/Theft/Robbery/Burglary of Health Supplies"]
missing = [c for c in UNIQUE_25_COLS if c not in atk25.columns]
if missing:
    warn(f"Columns not found in atk25: {missing}")
else:
    patch = atk25[UNIQUE_25_COLS].copy()
    patch.rename(columns={"Access Denied or Obstructed": "Access Denied or Obstructed_25"}, inplace=True)

    pre_cols = atk26.shape[1]
    atk26 = atk26.merge(patch, on="SiND Event ID", how="left")
    ok(f"atk26 cols: {pre_cols} → {atk26.shape[1]} (+{atk26.shape[1]-pre_cols} merged)")

    # If atk26 already has Access Denied column, consolidate
    if "Access Denied or Obstructed" not in atk26.columns and "Access Denied or Obstructed_25" in atk26.columns:
        atk26.rename(columns={"Access Denied or Obstructed_25": "Access Denied or Obstructed"}, inplace=True)
        ok("Renamed merged column to 'Access Denied or Obstructed'")

    merged_count = atk26["Looting/Theft/Robbery/Burglary of Health Supplies"].notna().sum()
    ok(f"{merged_count:,} rows received looting/theft values from atk25")


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Join food_systems into atk26 on Date + Admin 1
# ══════════════════════════════════════════════════════════════════════
log(f"{SEP}\nSTEP 5 — Join food_systems to atk26 on Date + Admin 1\n{SEP}")

# Ensure date columns are datetime
food["Date"]  = pd.to_datetime(food["Date"],  errors="coerce")
atk26["Date"] = pd.to_datetime(atk26["Date"], errors="coerce")

# Columns to bring from food (exclude already-shared cols)
SHARED = {"Date", "Country", "Country ISO", "Admin 1", "Latitude", "Longitude",
           "Geo Precision", "Reported Perpetrator", "Reported Perpetrator Name",
           "Weapon Carried/Used", "Date Event Entered", "Date Event Modified"}

food_cols = ["Date", "Admin 1"] + [c for c in food.columns if c not in SHARED]

# Aggregate food columns per day-region (multiple food incidents per day possible)
food_agg = (
    food[food_cols]
    .groupby(["Date", "Admin 1"])
    .agg(
        Food_Incidents_Count=("Food System Impact", "count"),
        Food_System_Impacts=("Food System Impact", lambda x: " | ".join(x.dropna().unique())),
        Food_Security_Categories=("All Food Security Categories", lambda x: " | ".join(x.dropna().unique())),
    )
    .reset_index()
)

pre_rows = len(atk26)
master = atk26.merge(food_agg, on=["Date", "Admin 1"], how="left")
matched = master["Food_Incidents_Count"].notna().sum()

ok(f"master rows: {pre_rows:,} → {len(master):,} (left join — no rows dropped)")
ok(f"{matched:,} health incidents matched to a food incident on same date + region")
ok(f"Match rate: {matched/len(master)*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — Clean insecurity: join food + prep text for NLP
# ══════════════════════════════════════════════════════════════════════
log(f"{SEP}\nSTEP 6 — Clean insecurity dataset\n{SEP}")

insecurity["Date"] = pd.to_datetime(insecurity["Date"], errors="coerce")

# 6a — Join food_systems onto insecurity (same key: Date + Admin 1)
insecurity_master = insecurity.merge(food_agg, on=["Date", "Admin 1"], how="left")
food_matched_ins  = insecurity_master["Food_Incidents_Count"].notna().sum()
ok(f"insecurity + food join: {len(insecurity_master):,} rows, {food_matched_ins:,} matched ({food_matched_ins/len(insecurity_master)*100:.1f}%)")

# 6b — Clean Event Description text for NLP
if "Event Description" in insecurity_master.columns:
    desc = insecurity_master["Event Description"].astype(str)

    # Strip leading date prefix (e.g. "28 December 2025: ") — redundant with Date col
    import re
    date_prefix = r"^\d{1,2}\s+\w+\s+\d{4}:\s*"
    insecurity_master["Event_Description_Clean"] = desc.str.replace(date_prefix, "", regex=True).str.strip()

    # Word count — useful for filtering very short/uninformative descriptions
    insecurity_master["Event_Description_Words"] = insecurity_master["Event_Description_Clean"].str.split().str.len()

    short = (insecurity_master["Event_Description_Words"] < 5).sum()
    ok(f"Event Description: cleaned date prefixes, added word count")
    ok(f"  {short} descriptions under 5 words (consider filtering for NLP)")
    ok(f"  Median length: {insecurity_master['Event_Description_Words'].median():.0f} words")

# 6c — Geo precision: flag rows with usable coordinates
if "Latitude" in insecurity_master.columns:
    insecurity_master["has_coords"] = insecurity_master["Latitude"].notna().astype(int)
    ok(f"has_coords flag: {insecurity_master['has_coords'].sum():,} rows with valid lat/lon")


# ══════════════════════════════════════════════════════════════════════
# BONUS — Add post_oct7 flag + year/month columns
# ══════════════════════════════════════════════════════════════════════
log(f"{SEP}\nBONUS — Feature engineering\n{SEP}")

for name, df in [("master", master), ("insecurity_master", insecurity_master), ("health", health), ("food", food)]:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"]       = df["Date"].dt.year
        df["Month"]      = df["Date"].dt.month
        df["post_oct7"]  = (df["Date"] >= "2023-10-07").astype(int)
        ok(f"{name}: added Year, Month, post_oct7")


# ══════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════
log(f"{SEP}\nSAVING CLEANED FILES → {OUT}\n{SEP}")

outputs = {
    "master_attacks_food_joined.csv":    master,
    "insecurity_master.csv":             insecurity_master,
    "health_2324_cleaned.csv":           health,
    "food_systems_cleaned.csv":          food,
    "attacks_1626_cleaned.csv":          atk26,
}

for filename, df in outputs.items():
    path = os.path.join(OUT, filename)
    df.to_csv(path, index=False)
    ok(f"Saved {filename}  ({len(df):,} rows × {df.shape[1]} cols)")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════
log(f"{SEP}\nSUMMARY\n{SEP}")

print(f"""
  master_attacks_food_joined.csv
    → attacks_1626 + food_systems joined on Date + Admin 1
    → {len(master):,} rows × {master.shape[1]} cols
    → Use for: time series (2016–2026), ML pipeline, pre/post Oct 7 comparison

  insecurity_master.csv
    → insecurity + food_systems joined  ← PRIMARY for NLP + geospatial work
    → {len(insecurity_master):,} rows × {insecurity_master.shape[1]} cols
    → Has: real lat/lon, full event text, cleaned descriptions, word counts
    → Use for: NLP (Project 3), geospatial mapping (Project 4)

  health_2324_cleaned.csv
    → HXL columns renamed, perpetrator standardized
    → {len(health):,} rows × {health.shape[1]} cols
    → Use for: health worker profession breakdown, SGBV analysis

  food_systems_cleaned.csv
    → Cleanest dataset, no nulls
    → {len(food):,} rows × {food.shape[1]} cols
    → Use for: food security standalone analysis

  attacks_1626_cleaned.csv
    → atk25 columns merged in, perpetrator standardized
    → {len(atk26):,} rows × {atk26.shape[1]} cols
    → Use for: longest time window, full categorical flags
""")

print("Done.")