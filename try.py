import pandas as pd
df = pd.read_csv("pse_topics_labeled.csv", parse_dates=["Date"])
df["ym"] = df["Date"].dt.strftime("%Y-%m")
print(df[df["ym"] >= "2025-08"].groupby("ym").size())