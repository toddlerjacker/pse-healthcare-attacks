"""
lethality_model.py
─────────────────────────────────────────────────────────────────────
Predicts whether an attack on healthcare was LETHAL, from circumstance
features only — then uses SHAP to show which factors drive lethality.

This is descriptive/explanatory ML: the goal is NOT forecasting, it is
identifying which attack characteristics are associated with a fatal
outcome, as an independent cross-check on the statistical findings.

CRITICAL — LEAKAGE CONTROL:
  Target Lethal_Incident == (Health Workers Killed > 0).
  Therefore every casualty-derived field is EXCLUDED as a predictor,
  or the model would simply re-read the answer. Features are limited to
  circumstances knowable independent of the casualty outcome:
  weapon, region, phase, perpetrator, target-type flags, timing.

Run AFTER feature2.py.  Reads pse_healthcare_featured_v2.csv.

Install once:
  pip install xgboost shap scikit-learn matplotlib --break-system-packages

Outputs:
  shap_summary.png        beeswarm — direction + magnitude per feature
  shap_bar.png            mean |SHAP| — ranked feature importance
  model_metrics.txt       baseline vs XGBoost, full classification report
  feature_importance.csv  ranked SHAP importances
─────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, f1_score
)
import xgboost as xgb
import shap

BASE = "/Users/andreisales/Desktop/document idf"
df = pd.read_csv(f"{BASE}/pse_healthcare_featured_v2.csv", parse_dates=["Date"])
print(f"Loaded {len(df):,} incidents")

# ── Target ────────────────────────────────────────────────────────────────────
y = df["Lethal_Incident"].astype(int)
print(f"Lethal incidents: {y.sum():,} ({y.mean()*100:.1f}%)  |  Non-lethal: {(1-y).sum():,}")

# ── LEAKAGE-SAFE FEATURE SET ──────────────────────────────────────────────────
# Categorical circumstances
cat_features = [
    "Weapon_Category",      # how the attack was carried out
    "Region",               # where
    "Conflict_Phase",       # when (phase)
    "Perpetrator_Simple",   # reported actor
]
# Binary circumstance / target-type flags (describe WHAT was attacked, not outcome)
bin_features = [
    "protected_entity",     # Red Cross / UN entity
    "residential_strike",   # struck a home/tent/residential building
    "children_affected",    # children referenced
    "repeat_target_text",   # textually-confirmed repeat strike
    "Facility_Attack",
    "Transport_Attack",
    "Access_Disruption",
    "is_weekend",
]
# Continuous (timing only — NOT casualty-derived)
num_features = [
    "days_since_oct7",
]

# Explicitly EXCLUDED (would leak the label):
#   Health Workers Killed/Injured/Arrested, Total_Casualties, Severity_*,
#   Has_Casualties, casualty_intensity, mass_casualty (partly casualty-derived),
#   targeting_score / targeting_label (contain mass_casualty term).

bin_features = [c for c in bin_features if c in df.columns]
num_features = [c for c in num_features if c in df.columns]

X_cat = pd.get_dummies(df[cat_features].astype(str), prefix=cat_features)
X = pd.concat([X_cat, df[bin_features].fillna(0), df[num_features].fillna(0)], axis=1)
X.columns = [str(c) for c in X.columns]
print(f"Feature matrix: {X.shape[1]} features")

# ── Split (stratified — preserve the 15% lethal rate in both sets) ────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

results = []

# ── Baseline: Logistic Regression ─────────────────────────────────────────────
# Establishes whether the problem needs a complex model at all.
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_train)
Xte_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=2000, class_weight="balanced")
lr.fit(Xtr_s, y_train)
lr_prob = lr.predict_proba(Xte_s)[:, 1]
lr_pred = (lr_prob >= 0.5).astype(int)

lr_auc = roc_auc_score(y_test, lr_prob)
lr_f1  = f1_score(y_test, lr_pred)
results.append(("Logistic Regression", lr_auc, lr_f1))
print(f"\nBaseline LR  — AUC {lr_auc:.3f} | F1 {lr_f1:.3f}")

# ── XGBoost ───────────────────────────────────────────────────────────────────
pos = int(y_train.sum()); neg = int((1 - y_train).sum())
scale_pos_weight = neg / pos    # counter class imbalance

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)
xgb_prob = model.predict_proba(X_test)[:, 1]
xgb_pred = (xgb_prob >= 0.5).astype(int)

xgb_auc = roc_auc_score(y_test, xgb_prob)
xgb_f1  = f1_score(y_test, xgb_pred)
results.append(("XGBoost", xgb_auc, xgb_f1))
print(f"XGBoost      — AUC {xgb_auc:.3f} | F1 {xgb_f1:.3f}")

# ── Write metrics report ──────────────────────────────────────────────────────
with open(f"{BASE}/model_metrics.txt", "w") as f:
    f.write("LETHALITY MODEL — leakage-controlled\n")
    f.write("="*55 + "\n")
    f.write(f"Incidents: {len(df):,}  |  Lethal: {y.sum():,} ({y.mean()*100:.1f}%)\n")
    f.write(f"Features: {X.shape[1]} (circumstance only; casualty fields excluded)\n\n")
    f.write(f"{'Model':<22}{'AUC':>8}{'F1':>8}\n")
    for name, auc, f1v in results:
        f.write(f"{name:<22}{auc:>8.3f}{f1v:>8.3f}\n")
    f.write("\nXGBoost classification report (threshold 0.5):\n")
    f.write(classification_report(y_test, xgb_pred,
            target_names=["Non-lethal", "Lethal"]))
    f.write("\nConfusion matrix [rows=true, cols=pred]:\n")
    f.write(str(confusion_matrix(y_test, xgb_pred)) + "\n")
print(f"\nSaved model_metrics.txt")

# ── SHAP ──────────────────────────────────────────────────────────────────────
print("Computing SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Beeswarm — shows direction (does the feature push toward lethal?)
plt.figure()
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.tight_layout()
plt.savefig(f"{BASE}/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# Bar — ranked mean |SHAP|
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=15, show=False)
plt.tight_layout()
plt.savefig(f"{BASE}/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved shap_summary.png and shap_bar.png")

# Ranked importance table
mean_abs = np.abs(shap_values).mean(axis=0)
imp = (pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs})
       .sort_values("mean_abs_shap", ascending=False))
imp.to_csv(f"{BASE}/feature_importance.csv", index=False)

print("\nTop 12 lethality drivers (mean |SHAP|):")
print(imp.head(12).to_string(index=False))
print("\nDone.")
