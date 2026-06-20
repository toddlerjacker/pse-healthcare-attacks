"""
feature_engineering_v2.py
Adds advanced targeting-analysis features to the PSE healthcare dataset.
Run AFTER feature_engineering.py.
Input:  pse_healthcare_featured.csv
Output: pse_healthcare_featured_v2.csv
"""

import pandas as pd
import numpy as np
import re

print("Loading featured dataset...")
df = pd.read_csv(
    "/Users/andreisales/Desktop/document idf/pse_healthcare_featured.csv",
    parse_dates=["Date"],
    low_memory=False
)
print(f"  Rows: {len(df):,}  |  Cols: {len(df.columns)}")

desc_col = "description_clean" if "description_clean" in df.columns else "Event Description"
df[desc_col] = df[desc_col].fillna("")

# ── 1. PROTECTED ENTITY FLAG ──────────────────────────────────────────────────
print("\n[1] Protected entity flag...")
kw_rc = df.get("kw_red_crescent", pd.Series(0, index=df.index))
kw_un = df.get("kw_un_staff",     pd.Series(0, index=df.index))
df["protected_entity"] = ((kw_rc == 1) | (kw_un == 1)).astype(int)
print(f"  Protected entity incidents: {df['protected_entity'].sum():,}")

# ── 2. RESIDENTIAL / FAMILY STRIKE ───────────────────────────────────────────
print("\n[2] Residential strike flag...")
df["residential_strike"] = df[desc_col].str.contains(
    r"\bhome\b|\bresidential\b|\bfamily\b|\bhouse\b|\btent\b|\bapartment\b",
    regex=True, flags=re.IGNORECASE
).astype(int)
print(f"  Residential strikes: {df['residential_strike'].sum():,}")

# ── 3. CHILDREN AFFECTED ─────────────────────────────────────────────────────
print("\n[3] Children affected flag...")
df["children_affected"] = df[desc_col].str.contains(
    r"\bchild(?:ren)?\b|\binfant\b|\bbaby\b|\bson\b|\bdaughter\b|\bkids?\b",
    regex=True, flags=re.IGNORECASE
).astype(int)
print(f"  Incidents mentioning children: {df['children_affected'].sum():,}")

# ── 4. MASS CASUALTY FLAG ────────────────────────────────────────────────────
print("\n[4] Mass casualty flag...")
df["mass_casualty_text"] = df[desc_col].str.contains(
    r"\bdozen|\bmultiple\b|\bmass\b|several\s+(?:killed|injured|dead)|\bnumerous\b",
    regex=True, flags=re.IGNORECASE
).astype(int)
df["mass_casualty"] = (
    (df["mass_casualty_text"] == 1) |
    (df.get("Total_Casualties", pd.Series(0, index=df.index)) >= 5)
).astype(int)
print(f"  Mass casualty incidents: {df['mass_casualty'].sum():,}")

# ── 5. TEXT-VERIFIED REPEAT TARGETING ────────────────────────────────────────
print("\n[5] Text-verified repeat targeting...")

repeat_pattern = (
    r'double.?tap|'
    r'\bsecond time\b|'
    r'\bthird time\b|'
    r'\bfourth time\b|'
    r'\bfifth time\b|'
    r'\brepeat(?:ed(?:ly)?)?\b|'
    r'\bbombed again\b|'
    r'\bstruck again\b|'
    r'\bhit again\b|'
    r'\btargeted again\b|'
    r'\battacked again\b|'
    r'\bonce again\b|'
    r'\balso (?:struck|hit|attack)|'
    r'\bprevious(?:ly)?\s+(?:attack|strike|hit|target|shell|bomb|raid)|'
    r'\bprevious attack\b|'
    r'\breturn(?:ed|ing)?\s+to\b|'
    r'\bearlier\s+(?:attack|strike|that day|raid|shelling|bombing)|'
    r'\bagain\s+(?:hit|struck|targeted|bombed|shelled|raided)|'
    r'\bonce more\b|'
    r'\byet again\b|'
    r'\bfor the (?:second|third|fourth|fifth) time\b'
)

df["repeat_target_text"] = df[desc_col].str.contains(
    repeat_pattern, regex=True, flags=re.IGNORECASE
).astype(int)
print(f"  Text-verified repeat targeting incidents: {df['repeat_target_text'].sum():,}")
print()

sub_patterns = [
    (r'double.?tap',                                          'Double-tap (strike on responders)'),
    (r'\bfor the (?:second|third|fourth|fifth) time\b|\bsecond time\b|\bthird time\b|\bfourth time\b', 'Nth time'),
    (r'\brepeat(?:ed(?:ly)?)?\b',                             'Repeatedly'),
    (r'\bprevious(?:ly)?\s+(?:attack|strike|hit|target|shell|bomb|raid)|\bprevious attack\b', 'Previous attack'),
    (r'\breturn(?:ed|ing)?\s+to\b',                           'Returned to'),
    (r'\bearlier\s+(?:attack|strike|that day|raid)',          'Earlier attack'),
    (r'(?:bombed|struck|hit|targeted|attacked) again|again (?:hit|struck|targeted|bombed)', 'Struck again'),
    (r'\balso (?:struck|hit|attack)',                         'Also struck/hit'),
]
for pat, label in sub_patterns:
    n = df[desc_col].str.contains(pat, regex=True, flags=re.IGNORECASE).sum()
    if n > 0:
        print(f"    {label}: {n}")
# ── 6. TIME-BASED FEATURES ───────────────────────────────────────────────────
print("\n[6] Time-based features...")
df["days_since_oct7"] = (df["Date"] - pd.Timestamp("2023-10-07")).dt.days

df_sorted = df.sort_values("Date")
df["days_since_last_attack"] = df_sorted["Date"].diff().dt.days.reindex(df.index)

df["is_weekend"] = df["Date"].dt.dayofweek.isin([4, 5]).astype(int)
print(f"  Days since Oct 7 range: {df['days_since_oct7'].min()} to {df['days_since_oct7'].max()}")

# ── 7. ESCALATION VELOCITY (monthly) ─────────────────────────────────────────
print("\n[7] Escalation velocity...")
monthly_counts = df.groupby("Year_Month").size().reset_index(name="month_count")
monthly_counts["month_velocity_pct"] = monthly_counts["month_count"].pct_change() * 100
df = df.merge(monthly_counts[["Year_Month", "month_count", "month_velocity_pct"]],
              on="Year_Month", how="left")
print(f"  Max month-over-month increase: {monthly_counts['month_velocity_pct'].max():.0f}%")

# ── 8. SYSTEMATIC TARGETING SCORE ────────────────────────────────────────────
print("\n[8] Systematic targeting score...")
gunfire_flag = (
    df.get("Weapon_Category", pd.Series("", index=df.index)) == "Gunfire/Small Arms"
).astype(int)

df["targeting_score"] = (
    df["protected_entity"]   * 3 +
    df["repeat_target_text"] * 3 +
    df["mass_casualty"]      * 1
)

df["targeting_label"] = pd.cut(
    df["targeting_score"],
    bins=[-0.1, 0, 2, 4, 99],
    labels=["None", "Low", "Moderate", "High"]
).astype(str)
print(f"  Targeting label distribution:\n{df['targeting_label'].value_counts().to_string()}")
df["targeting_map"] = (df["targeting_score"] >= 3).map({True: "Targeted", False: "Not flagged"})
# ── 9. LETHALITY / INTENSITY ─────────────────────────────────────────────────
print("\n[9] Per-incident lethality...")
df["casualty_intensity"] = (
    df["Health Workers Killed"].fillna(0) +
    df["Health Workers Injured"].fillna(0)
)

# ── SAVE ─────────────────────────────────────────────────────────────────────
print("\n[10] Saving...")
df = df.drop(columns=["mass_casualty_text"], errors="ignore")

out_path = "/Users/andreisales/Desktop/document idf/pse_healthcare_featured_v2.csv"
df.to_csv(out_path, index=False)

print(f"\n Done!")
print(f"   Rows: {len(df):,}")
print(f"   Cols: {len(df.columns)}")
print(f"   Saved to: {out_path}")
print(f"\nNew features added:")
for c in ["protected_entity", "residential_strike", "children_affected",
          "mass_casualty", "repeat_target_text", "days_since_oct7",
          "days_since_last_attack", "is_weekend", "month_count",
          "month_velocity_pct", "targeting_score", "targeting_label",
          "casualty_intensity"]:
    print(f"   + {c}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("KEY INSIGHTS FROM NEW FEATURES")
print("="*60)
print(f"Protected entity attacks:           {df['protected_entity'].sum():,}")
print(f"Residential/family strikes:         {df['residential_strike'].sum():,}")
print(f"Incidents mentioning children:      {df['children_affected'].sum():,}")
print(f"Mass casualty events:               {df['mass_casualty'].sum():,}")
print(f"Text-verified repeat targeting:     {df['repeat_target_text'].sum():,}")
print(f"High targeting score:               {(df['targeting_label'] == 'High').sum():,}")
print(f"Moderate targeting score:           {(df['targeting_label'] == 'Moderate').sum():,}")
print()
print("Note: repeat_target_text verified from description language only.")
print("      (double-tap, second/third/fourth time, repeatedly, etc.)")