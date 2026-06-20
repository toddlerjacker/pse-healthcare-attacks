import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


file_path = "2016-2026-pse-attacks-on-health-care-incident-data.xlsx"
df = pd.read_excel(file_path)

# Convert Date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Filter 2023–2026
df_2023_2026 = df[(df["Date"] >= "2023-01-01") & (df["Date"] <= "2026-12-31")].copy()

# Search for "israel" in perpetrator name, ignoring capitals
israel_mask = df_2023_2026["Reported Perpetrator Name"].astype(str).str.contains(
    "israel", case=False, na=False
)

df_israel = df_2023_2026[israel_mask].copy()

# Monthly totals
all_monthly = df_2023_2026.groupby(pd.Grouper(key="Date", freq="MS")).size()
israel_monthly = df_israel.groupby(pd.Grouper(key="Date", freq="MS")).size()

# Combine into one table
monthly_compare = pd.DataFrame({
    "All Attacks": all_monthly,
    "Attacks by Israel Forces/Proxies'": israel_monthly
}).fillna(0)

monthly_compare["All Attacks"] = monthly_compare["All Attacks"].astype(int)
monthly_compare["Attacks by Israel Forces/Proxies'"] = monthly_compare["Attacks by Israel Forces/Proxies'"].astype(int)

# Optional: percentage of monthly attacks involving Israel
monthly_compare["Percent with Israel"] = (
    monthly_compare["Attacks by Israel Forces/Proxies'"] / monthly_compare["All Attacks"] * 100
).round(1)

print(monthly_compare)

# Graph
plt.figure(figsize=(12, 6))

plt.bar(
    monthly_compare.index,
    monthly_compare["Attacks by Israel Forces/Proxies'"],
    width=20,
    label="Attacks by Israel Forces/Proxies'"
)

plt.title("Monthly Attacks (2023–2026) by Israel Forces/Proxies")
plt.xlabel("Month")
plt.ylabel("Number of incidents")

plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("monthly_attacks_israel_2023_2026_bar.png")
plt.show()


weapon_col = df_2023_2026["Weapon Carried/Used"].fillna("").astype(str)

def has_token(series, token):
    pattern = rf"(^|,\s*){token}($|,)"
    return series.str.contains(pattern, case=False, na=False, regex=True)

weapon_masks = {
    "Aerial Bomb": weapon_col.str.contains(r"Aerial Bomb", case=False, na=False, regex=True),
    "Hand Grenade": has_token(weapon_col, "Hand Grenade"),
    "Missile": has_token(weapon_col, "Missile"),
    "Rocket": has_token(weapon_col, "Rocket"),
    "Unspecified Explosive": has_token(weapon_col, "Unspecified Explosive"),
    "Unspecified IED": has_token(weapon_col, "Unspecified IED"),
}

monthly_counts = pd.DataFrame()

for weapon, mask in weapon_masks.items():
    counts = (
        df_2023_2026[mask]
        .groupby(pd.Grouper(key="Date", freq="ME"))
        .size()
    )
    monthly_counts[weapon] = counts

monthly_counts = monthly_counts.fillna(0)

plt.figure(figsize=(14, 7))
for col in monthly_counts.columns:
    plt.plot(monthly_counts.index, monthly_counts[col], label=col)

plt.title("Monthly Incidents by Weapon Type (2023–2026)")
plt.xlabel("Month")
plt.ylabel("Number of incidents")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

df.head()