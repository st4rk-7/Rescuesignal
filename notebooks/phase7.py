# Step 7.2: THE final test, run once. Settings frozen in step 7.1 (CLAUDE.md, 2026-10-29).
# Train on all 11 dev events, test on the 2 locked events. Nothing here may be changed after seeing the scores.
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             classification_report, ConfusionMatrixDisplay)
from phase6_combine import make_pipeline

OUT = Path("../results/final"); OUT.mkdir(exist_ok=True)
TEST_EVENTS = ["cyclone_idai_2019", "kaikoura_earthquake_2016"]   # locked in step 4.1
df = pd.read_csv("../data/set1_clean.tsv", sep="\t")
train, test = df[~df["event"].isin(TEST_EVENTS)], df[df["event"].isin(TEST_EVENTS)]

VARIANTS = {"FINAL_C_both": (True, True), "A_tfidf": (True, False), "B_feats": (False, True)}   # A, B = ablation only
rows, preds = [], {}
for variant, (t, f) in VARIANTS.items():
    pipe = make_pipeline(LinearSVC(C=0.1, class_weight="balanced"), t, f).fit(train[["tweet_text"]], train["class_label"])
    for event in TEST_EVENTS + ["both"]:
        part = test if event == "both" else test[test["event"] == event]
        y, pred = part["class_label"], pipe.predict(part[["tweet_text"]])
        present = sorted(y.unique())
        rows.append({"variant": variant, "event": event, "n": len(y),
                     "macro_f1": f1_score(y, pred, labels=present, average="macro"),
                     "accuracy": accuracy_score(y, pred),
                     "macro_p": precision_score(y, pred, labels=present, average="macro", zero_division=0),
                     "macro_r": recall_score(y, pred, labels=present, average="macro", zero_division=0),
                     "weighted_f1": f1_score(y, pred, average="weighted")})
        if variant == "FINAL_C_both":
            preds[event] = (y, pred)

res = pd.DataFrame(rows)
res.to_csv(OUT / "final_scores.csv", index=False)
print(res.round(3).to_string(index=False))

final = test.assign(pred=preds["both"][1])                          # saved for the error analysis in 7.3
final.to_csv(OUT / "final_predictions.tsv", sep="\t", index=False)
for event in TEST_EVENTS:
    y, pred = preds[event]
    print(f"\n=== {event} (final model) ===")
    print(classification_report(y, pred, zero_division=0, digits=3))
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(f"{event}: final model (true rows, predicted columns)")
    fig.tight_layout(); fig.savefig(OUT / f"confusion_{event}.png", dpi=120); plt.close(fig)
