"""
Dataset Analysis Script
Analyzes all four PSE/Gaza conflict datasets and outputs a full profile.
Run: python analyze_datasets.py
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
FILES = {
    "food_systems":   "/Users/andreisales/Desktop/document idf/2023-2025-pse-gaza-conflict-incidents-affecting-food-systems-incident-data-incident-data.xlsx",
    "health_2324":    "/Users/andreisales/Desktop/document idf/2023-2024-pse-shcc-health-care-data.xlsx",
    "attacks_1626":   "/Users/andreisales/Desktop/document idf/2016-2026-pse-attacks-on-health-care-incident-data.xlsx",
    "attacks_1625":   "/Users/andreisales/Desktop/document idf/2016-2025-pse-attacks-on-health-care-incident-data.xlsx",
    "insecurity": "/Users/andreisales/Desktop/document idf/InsecurityInsight_AttacksonHealthcareOPT (shared on 06.03.26).xlsx"
}

SEP = "=" * 70

# ── HELPERS ───────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def subsection(title):
    print(f"\n--- {title} ---")

def load_file(label, path):
    """Load xlsx, auto-detecting header row (skips HXL tag rows)."""
    if not os.path.exists(path):
        print(f"  [SKIP] File not found: {path}")
        return None

    size_kb = os.path.getsize(path) / 1024
    print(f"  Loading {label} ({size_kb:.1f} KB)...")

    # Try header=0 first; if first data row looks like HXL tags, use header=1
    df = pd.read_excel(path, sheet_name=0)
    first_row = df.iloc[0].astype(str).str.contains(r"^#", na=False)
    if first_row.sum() > 3:
        df = pd.read_excel(path, sheet_name=0, header=1)

    print(f"  → {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def dtype_summary(df):
    """Summarize column types."""
    types = df.dtypes.value_counts()
    return ", ".join(f"{str(k).split('[')[0]}({v})" for k, v in types.items())


def date_range(df):
    """Find and return the date range from any date-like column."""
    for col in df.columns:
        if "date" in str(col).lower():
            parsed = pd.to_datetime(df[col], errors="coerce")
            valid = parsed.dropna()
            if len(valid) > 10:
                return col, valid.min().date(), valid.max().date(), len(valid)
    return None, None, None, 0


def numeric_profile(df):
    """Profile all numeric columns: sum, mean, nulls, non-zero count."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    # Exclude IDs and coords
    exclude = ["id", "lat", "lon", "latitude", "longitude", "iso", "event"]
    num_cols = [c for c in num_cols if not any(x in c.lower() for x in exclude)]

    if not num_cols:
        print("  No numeric columns found.")
        return

    rows = []
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "Column": col[:55],
            "Sum": f"{s.sum():,.0f}",
            "Mean": f"{s.mean():.2f}",
            "Max": f"{s.max():,.0f}",
            "Non-Zero": f"{(s > 0).sum():,}",
            "Nulls": f"{s.isna().sum():,}",
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))


def categorical_profile(df, top_n=8):
    """Profile categorical columns with low-to-mid cardinality."""
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    skip_keywords = ["description", "name", "notes", "outcome", "id", "event"]

    for col in cat_cols:
        if any(k in col.lower() for k in skip_keywords):
            continue
        n_unique = df[col].nunique()
        if n_unique < 2 or n_unique > 60:
            continue
        vc = df[col].value_counts(dropna=False).head(top_n)
        print(f"\n  {col!r}  ({n_unique} unique values)")
        for val, cnt in vc.items():
            pct = cnt / len(df) * 100
            bar = "█" * int(pct / 2)
            print(f"    {str(val)[:40]:<42} {cnt:>5,}  {pct:5.1f}%  {bar}")


def null_report(df, threshold=0.3):
    """Flag columns with >30% nulls."""
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > threshold].sort_values(ascending=False)
    if high_null.empty:
        print("  No columns above 30% null threshold.")
    else:
        for col, pct in high_null.items():
            print(f"  {col[:60]:<62} {pct*100:.1f}% null")


def overlap_check(dfs):
    """Check column overlap across all loaded datasets."""
    section("CROSS-DATASET COLUMN OVERLAP")
    all_cols = {label: set(df.columns.str.lower().str.strip())
                for label, df in dfs.items()}
    labels = list(all_cols.keys())

    # Shared across ALL
    common_all = set.intersection(*all_cols.values())
    print(f"\nShared across ALL {len(labels)} datasets ({len(common_all)} columns):")
    for c in sorted(common_all):
        print(f"  • {c}")

    # Pairwise
    print("\nPairwise shared columns:")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            shared = all_cols[labels[i]] & all_cols[labels[j]]
            print(f"  {labels[i]} ∩ {labels[j]}: {len(shared)} shared")

    # Unique to each
    print("\nColumns unique to each dataset:")
    for label, cols in all_cols.items():
        others = set.union(*[v for k, v in all_cols.items() if k != label])
        unique = cols - others
        print(f"  {label} ({len(unique)} unique): {', '.join(sorted(unique)[:10])}"
              + ("..." if len(unique) > 10 else ""))


def combined_stats(dfs):
    """Cross-dataset summary of key casualty/incident metrics."""
    section("COMBINED METRICS SUMMARY")

    casualty_keywords = ["killed", "injured", "wounded", "dead", "death",
                         "kidnapped", "arrested", "threatened", "assaulted"]
    facility_keywords = ["destruction", "damaged", "destroyed", "facility"]

    print(f"\n{'Dataset':<15} {'HW Killed':>12} {'HW Injured':>12} {'Facility Destroyed':>20} {'Facility Damaged':>18}")
    print("-" * 80)

    for label, df in dfs.items():
        def col_sum(keywords):
            for col in df.columns:
                if all(k in col.lower() for k in keywords):
                    return pd.to_numeric(df[col], errors="coerce").sum()
            return 0

        killed   = col_sum(["killed"])
        injured  = col_sum(["injured"])
        dest     = col_sum(["destruction"])
        damaged  = col_sum(["damaged"])
        print(f"{label:<15} {killed:>12,.0f} {injured:>12,.0f} {dest:>20,.0f} {damaged:>18,.0f}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def analyze():
    print(f"\n{'█'*70}")
    print("  PSE / GAZA CONFLICT DATASET ANALYSIS")
    print(f"{'█'*70}")

    section("LOADING FILES")
    dfs = {}
    for label, path in FILES.items():
        df = load_file(label, path)
        if df is not None:
            dfs[label] = df

    if not dfs:
        print("\n[ERROR] No files loaded. Check your file paths.")
        return

    # ── Per-dataset analysis ──────────────────────────────────────────────────
    for label, df in dfs.items():
        section(f"DATASET: {label.upper()}")

        # Basic info
        subsection("Shape & Types")
        print(f"  Rows:    {df.shape[0]:,}")
        print(f"  Columns: {df.shape[1]}")
        print(f"  Types:   {dtype_summary(df)}")

        date_col, dmin, dmax, n_dates = date_range(df)
        if date_col:
            print(f"  Date col: '{date_col}'  →  {dmin} to {dmax}  ({n_dates:,} valid dates)")

        # All columns
        subsection("All Columns")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype).split("[")[0]
            n_null = df[col].isna().sum()
            n_uniq = df[col].nunique()
            print(f"  {i:>3}. {col[:55]:<57} {dtype:<12} "
                  f"nulls={n_null:<6,} unique={n_uniq:,}")

        # Numeric profile
        subsection("Numeric Columns — Key Statistics")
        numeric_profile(df)

        # Categorical profile
        subsection("Categorical Columns — Value Distributions")
        categorical_profile(df)

        # Null report
        subsection("Data Quality — High-Null Columns (>30%)")
        null_report(df)

        # Sample rows
        subsection("Sample Data (3 rows)")
        print(df.head(3).to_string())

    # ── Cross-dataset ─────────────────────────────────────────────────────────
    if len(dfs) > 1:
        overlap_check(dfs)
        combined_stats(dfs)

    section("DONE")
    print(f"  Analyzed {len(dfs)} dataset(s) successfully.\n")


if __name__ == "__main__":
    analyze()