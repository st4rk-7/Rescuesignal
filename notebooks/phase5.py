# Step 5.1: compare 5 classical models with the same 11 leave-one-event-out folds (default settings).
import sys, time
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from phase2 import clean

TEST_EVENTS = ["cyclone_idai_2019", "kaikoura_earthquake_2016"]   # locked in step 4.1
df = pd.read_csv("../data/set1_clean.tsv", sep="\t")
dev = df[~df["event"].isin(TEST_EVENTS)].reset_index(drop=True)

MODELS = {
    "LogReg":       lambda: LogisticRegression(max_iter=1000),
    "LinearSVM":    lambda: LinearSVC(),
    "MultinomialNB": lambda: MultinomialNB(),
    "DecisionTree": lambda: DecisionTreeClassifier(random_state=42),
    "RandomForest": lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    # Step 5.2: same models, but mistakes on rare labels cost more (MultinomialNB has no class_weight)
    "LogReg+bal":       lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "LinearSVM+bal":    lambda: LinearSVC(class_weight="balanced"),
    "DecisionTree+bal": lambda: DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "RandomForest+bal": lambda: RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
}

def run_fold(name, train_idx, val_idx):
    tr, va = dev.iloc[train_idx], dev.iloc[val_idx]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("model", MODELS[name]()),
    ]).fit(tr["tweet_text"], tr["class_label"])
    y, pred = va["class_label"], pipe.predict(va["tweet_text"])
    present = sorted(y.unique())                       # score only labels this event contains
    return {"model": name, "event": va["event"].iloc[0],
            "macro_f1": f1_score(y, pred, labels=present, average="macro"),
            "macro_p": precision_score(y, pred, labels=present, average="macro", zero_division=0),
            "macro_r": recall_score(y, pred, labels=present, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, average="weighted"),
            "accuracy": accuracy_score(y, pred),
            **{f"recall:{c}": recall_score(y, pred, labels=[c], average="macro", zero_division=0) for c in present}}

folds = list(LeaveOneGroupOut().split(dev, groups=dev["event"]))
from pathlib import Path
CACHE = Path("../results/model_runs"); CACHE.mkdir(exist_ok=True)
for name in MODELS:
    out = CACHE / f"{name}.csv"
    if out.exists():                                  # already run: a crash later doesn't lose it
        continue
    t = time.time()
    jobs = 3 if "Forest" in name else 11              # forests are memory-heavy: fewer at once
    pd.DataFrame(Parallel(n_jobs=jobs)(delayed(run_fold)(name, a, b) for a, b in folds)).to_csv(out, index=False)
    print(f"{name} done in {time.time() - t:.0f}s", flush=True)

res = pd.concat([pd.read_csv(CACHE / f"{n}.csv") for n in MODELS], ignore_index=True)
res.to_csv("../results/models_loeo.csv", index=False)
summary = res.drop(columns="event").groupby("model").agg(["mean", "std"])  # recall:<label> NaN where absent, ignored
table = pd.DataFrame({m: summary[m]["mean"].map("{:.3f}".format) + " ± " + summary[m]["std"].map("{:.3f}".format)
                      for m in ["macro_f1", "accuracy", "macro_p", "macro_r", "weighted_f1"]})
print(table.sort_values("macro_f1", ascending=False).to_string())

recall = res.filter(like="recall:").groupby(res["model"]).mean().T     # mean recall per label over folds that contain it
recall.index = [i.split(":")[1] for i in recall.index]
print("\nper-label recall (mean over events that contain the label):")
print(recall.round(2).to_string())
