import pandas as pd
import numpy as np
from folium.plugins import HeatMap
from folium import LayerControl

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("/Users/andreisales/Desktop/document idf/pse_jittered.csv",
                 parse_dates=["Date"])
print(f"Total rows: {len(df)}")


# If collateral, health worker deaths = proportional to their % of population
# Health workers are ~0.5% of Gaza population
# If their death rate is >> 0.5% of total deaths → targeted
hw_killed = df["Health Workers Killed"].sum()
print(f"Total health workers killed: {hw_killed}")
# Compare to OCHA total civilian death count for same period
# Ratio tells the story

protected = df[(df["kw_red_crescent"] == 1) | (df["kw_un_staff"] == 1)]
print(f"Attacks on explicitly protected entities: {len(protected)}")
print(protected["Weapon_Category"].value_counts())
print(protected["Health Workers Killed"].sum(), "killed")

from scipy.stats import poisson
# Under null hypothesis (random): HW deaths ~ Poisson(expected)
expected = 45000 * 0.005  # 225 expected if truly proportional
observed = 761
p_value = 1 - poisson.cdf(observed - 1, expected)
print(f"P-value (one-tailed): {p_value:.2e}")
# Will be astronomically small — effectively 0

# If attacks were accidental collateral damage, they'd be random
# Test: are attacks clustered in time beyond what's expected?
import scipy.stats as stats

daily_counts = df.groupby("Date").size()
# Run a dispersion test — variance >> mean = clustering
mean, var = daily_counts.mean(), daily_counts.var()
dispersion_index = var / mean
print(f"Dispersion index: {dispersion_index:.2f}")
# D > 1 = clustered, D = 1 = random (Poisson), D < 1 = uniform
# Collateral damage would look random (D ≈ 1)
# Systematic targeting would be clustered (D >> 1)


print("Raw values bucketed into Perpetrator 'Other':")
print(df[df["Perpetrator_Simple"] == "Other"]["Reported Perpetrator Name"].value_counts())

print()
print("Raw values bucketed into Perpetrator 'Unknown':")
print(df[df["Perpetrator_Simple"] == "Unknown"]["Reported Perpetrator Name"].value_counts())

print()
print("Raw values bucketed into Weapon 'Other':")
print(df[df["Weapon_Category"] == "Other"]["Weapon Carried/Used"].value_counts())

print()
print("ALL unique raw perpetrator name values:")
print(df["Reported Perpetrator Name"].value_counts())