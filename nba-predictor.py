import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset
df = pd.read_csv("matches.csv", index_col=0, parse_dates=["date"])

# Feature engineering
df["h/a"] = (df["venue"] == "Home").astype(int)
df["opp"] = df["opponent"].astype("category").cat.codes
df["hour"] = df["time"].str.replace(":.+", "", regex=True).astype(float)
df["day"] = df["date"].dt.dayofweek
df["target"] = (df["result"] == "W").astype(int)

# Rolling statistics per team
ROLL_N = 5
def add_roll(g):
    g = g.sort_values("date")
    for col in ["pf", "pa"]:
        g[f"{col}_rolling_{ROLL_N}"] = (
            g[col].shift(1).rolling(ROLL_N, min_periods=ROLL_N).mean()
        )
    g[f"pdiff_rolling_{ROLL_N}"] = g[f"pf_rolling_{ROLL_N}"] - g[f"pa_rolling_{ROLL_N}"]
    return g

df = df.groupby("team", group_keys=False).apply(add_roll)
need_cols = [f"pf_rolling_{ROLL_N}", f"pa_rolling_{ROLL_N}", f"pdiff_rolling_{ROLL_N}"]
df = df.dropna(subset=need_cols)

# Time-based train/test split (80/20)
cutoff = df["date"].quantile(0.80)
train = df[df["date"] <= cutoff]
test  = df[df["date"] >  cutoff]

predictors = [
    "h/a", "opp", "hour", "day",
    f"pf_rolling_{ROLL_N}", f"pa_rolling_{ROLL_N}", f"pdiff_rolling_{ROLL_N}"
]

# Train model
rf = RandomForestClassifier(
    n_estimators=400,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(train[predictors], train["target"])

# Evaluate
preds = rf.predict(test[predictors])
probs = rf.predict_proba(test[predictors])[:, 1]

acc = accuracy_score(test["target"], preds)
prec = precision_score(test["target"], preds)
rec = recall_score(test["target"], preds)
cm = confusion_matrix(test["target"], preds)

print("Rows -> Train:", len(train), "| Test:", len(test), "| Cutoff:", cutoff.date())
print("accuracy: ", round(acc, 3))
print("precision:", round(prec, 3))
print("recall:   ", round(rec, 3))
print("confusion matrix:\n", cm)

# Feature importances
fi = pd.Series(rf.feature_importances_, index=predictors).sort_values(ascending=False)
print("\nFeature importances:")
print(fi.to_string())

# Predictions table
out_cols = ["date", "team", "opponent", "venue", "pf", "pa", "result"]
pred_table = test[out_cols].copy()
pred_table["pred_win"] = preds
pred_table["prob_win"] = probs.round(3)

pred_table.sort_values(["date","team"]).to_csv("nba_predictions.csv", index=False)
print("\nSample predictions:")
print(pred_table.sort_values("date").head(10).to_string(index=False))
print("\nSaved predictions to: nba_predictions.csv")
