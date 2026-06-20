import pandas as pd

df = pd.read_csv("/Users/andreisales/Desktop/document idf/pse_healthcare_master_clean.csv")

# Find rows where description mentions more deaths than recorded
high_killed = df[df["Health Workers Killed"] > 5].sort_values("Health Workers Killed", ascending=False)
print(high_killed[["Date", "Health Workers Killed", "description_clean"]].head(10).to_string())