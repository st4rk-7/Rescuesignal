# Step 5.3: one results table, 5 models, mean ± SD over the 11 leave-one-event-out folds.
# Reads the saved runs from phase5.py (no retraining).
import pandas as pd

# Proposal decision: class_weight="balanced" for LR/SVM/DT/RF; NB has no class_weight.
CHOSEN = {"LinearSVM+bal": "LinearSVM (balanced)", "LogReg+bal": "LogReg (balanced)", "RandomForest+bal": "RandomForest (balanced)",
          "DecisionTree+bal": "DecisionTree (balanced)", "MultinomialNB": "MultinomialNB (no weights)"}
METRICS = {"macro_f1": "Macro-F1", "accuracy": "Accuracy", "macro_p": "Macro-P",
           "macro_r": "Macro-R", "weighted_f1": "Weighted-F1"}

res = pd.read_csv("../results/models_loeo.csv")
res = res[res["model"].isin(CHOSEN)]
assert res.groupby("model").size().eq(11).all(), "every model needs all 11 folds"

stats = res.groupby("model")[list(METRICS)].agg(["mean", "std"])
table = pd.DataFrame({name: stats[m]["mean"].map("{:.3f}".format) + " ± " + stats[m]["std"].map("{:.3f}".format)
                      for m, name in METRICS.items()})
table = table.loc[stats["macro_f1"]["mean"].sort_values(ascending=False).index].rename(index=CHOSEN)
table.index.name = "Model"
table.to_csv("../results/table_5_3_models.csv")
print(table.to_markdown())
