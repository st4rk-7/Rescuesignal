# Step 5.4: small grid (C x min_df) for the top 2 models, on the 11 dev folds only (test events stay locked).
import itertools, time
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from phase2 import clean

TEST_EVENTS = ["cyclone_idai_2019", "kaikoura_earthquake_2016"]   # locked in step 4.1
df = pd.read_csv("../data/set1_clean.tsv", sep="\t")
dev = df[~df["event"].isin(TEST_EVENTS)].reset_index(drop=True)
folds = list(LeaveOneGroupOut().split(dev, groups=dev["event"]))

MODELS = {"LinearSVM+bal": lambda C: LinearSVC(C=C, class_weight="balanced"),
          "LogReg+bal":    lambda C: LogisticRegression(C=C, max_iter=1000, class_weight="balanced")}
GRID = list(itertools.product(MODELS, [0.1, 1.0, 10.0], [2, 5]))   # (model, C, min_df)

def run_fold(name, C, min_df, train_idx, val_idx):
    tr, va = dev.iloc[train_idx], dev.iloc[val_idx]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
        ("model", MODELS[name](C)),
    ]).fit(tr["tweet_text"], tr["class_label"])
    y, pred = va["class_label"], pipe.predict(va["tweet_text"])
    return {"model": name, "C": C, "min_df": min_df, "event": va["event"].iloc[0],
            "macro_f1": f1_score(y, pred, labels=sorted(y.unique()), average="macro")}   # same metric as 5.1

t = time.time()
res = pd.DataFrame(Parallel(n_jobs=11)(delayed(run_fold)(n, C, m, a, b) for n, C, m in GRID for a, b in folds))
res.to_csv("../results/tuning_5_4.csv", index=False)
print(f"done in {time.time() - t:.0f}s")

summary = res.groupby(["model", "C", "min_df"])["macro_f1"].agg(["mean", "std"]).round(3)
print(summary.to_string())
# self-check: C=1, min_df=2 is exactly the step 5.2 setting, so it must reproduce 0.641 / 0.629
print("\nreproduces 5.2:", summary.loc[("LinearSVM+bal", 1.0, 2), "mean"], summary.loc[("LogReg+bal", 1.0, 2), "mean"])
