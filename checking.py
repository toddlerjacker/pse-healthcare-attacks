import pandas as pd
import re

df = pd.read_csv("/Users/andreisales/Desktop/document idf/pse_healthcare_featured_v2.csv",
                 low_memory=False)

repeat_patterns = [
    r'double.?tap',
    r'second strike',
    r'struck again',
    r'hit again',
    r'targeted again',
    r'attacked again',
    r'previously (attack|struck|hit|target)',
    r'same (facility|hospital|location|building|site|ambulance)',
    r'another strike',
    r'another attack',
    r'once again',
    r'\bagain\b',
    r'second time',
    r'third time',
    r'repeated(ly)?',
    r'follow.?up (strike|attack)',
    r'also (struck|hit|attack)',
    r'return(ed|ing) to',
]

desc = df["description_clean"].fillna("")

print(f"Total rows: {len(df)}")
print()

for pat in repeat_patterns:
    matches = df[desc.str.contains(pat, case=False, regex=True)]
    if len(matches) > 0:
        print(f"=== {pat} ({len(matches)} hits) ===")
        for _, row in matches.head(3).iterrows():
            print(f"  {str(row['description_clean'])[:160]}")
        print()