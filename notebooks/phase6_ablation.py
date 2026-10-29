# Step 6.4: ablation. (A) TF-IDF only, (B) 25 features only, (C) both, for the 5 models, same 11 dev folds.
# Model settings = step 5.3 table (balanced for LR/SVM/DT/RF, default C), so A reproduces 5.3.
import time
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from phase6_combine import make_pipeline

TEST_EVENTS = ["cyclone_idai_2019", "kaikoura_earthquake_2016"]   # locked in step 4.1
df = pd.read_csv("../data/set1_clean.tsv", sep="\t")
dev = df[~df["event"].isin(TEST_EVENTS)].reset_index(drop=True)
folds = list(LeaveOneGroupOut().split(dev, groups=dev["event"]))

MODELS = {
    "LinearSVM":     lambda: LinearSVC(class_weight="balanced"),
    "LogReg":        lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "MultinomialNB": lambda: MultinomialNB(),
    "DecisionTree":  lambda: DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "RandomForest":  lambda: RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
}
VARIANTS = {"A_tfidf": (True, False), "B_feats": (False, True), "C_both": (True, True)}

def run_fold(model, variant, train_idx, val_idx):
    tr, va = dev.iloc[train_idx], dev.iloc[val_idx]
    pipe = make_pipeline(MODELS[model](), *VARIANTS[variant]).fit(tr[["tweet_text"]], tr["class_label"])
    y, pred = va["class_label"], pipe.predict(va[["tweet_text"]])
    present = sorted(y.unique())                       # score only labels this event contains
    return {"model": model, "variant": variant, "event": va["event"].iloc[0],
            "macro_f1": f1_score(y, pred, labels=present, average="macro"),
            **{f"f1:{c}": f1_score(y, pred, labels=[c], average="macro", zero_division=0) for c in present}}

CACHE = Path("../results/ablation_runs"); CACHE.mkdir(exist_ok=True)
for model in MODELS:
    for variant in VARIANTS:
        out = CACHE / f"{model}_{variant}.csv"
        if out.exists():                              # already run: a crash later doesn't lose it
            continue
        t = time.time()
        jobs = 3 if "Forest" in model else 11         # forests are memory-heavy: fewer at once
        pd.DataFrame(Parallel(n_jobs=jobs)(delayed(run_fold)(model, variant, a, b) for a, b in folds)).to_csv(out, index=False)
        print(f"{model} {variant} done in {time.time() - t:.0f}s", flush=True)

res = pd.concat([pd.read_csv(CACHE / f"{m}_{v}.csv") for m in MODELS for v in VARIANTS], ignore_index=True)
res.to_csv("../results/ablation_6_4.csv", index=False)

stats = res.groupby(["model", "variant"])["macro_f1"].agg(["mean", "std"])
table = (stats["mean"].map("{:.3f}".format) + " ± " + stats["std"].map("{:.3f}".format)).unstack("variant")
print("\nMacro-F1, mean ± SD over 11 dev events:")
print(table.to_string())

# per-label F1 change C - A (does adding the features help a label?), mean over events that contain it
perlabel = res.filter(like="f1:").groupby([res["model"], res["variant"]]).mean()
delta = (perlabel.xs("C_both", level="variant") - perlabel.xs("A_tfidf", level="variant")).T
delta.index = [i.split(":")[1] for i in delta.index]
print("\nper-label F1 change, C (both) minus A (TF-IDF only):")
print(delta.round(3).to_string())
