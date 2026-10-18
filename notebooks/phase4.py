# Step 4.2: leave-one-event-out on the 11 development events.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score, accuracy_score
from phase2 import clean

TEST_EVENTS = ["cyclone_idai_2019", "kaikoura_earthquake_2016"]   # locked in step 4.1, used only in Phase 7

df = pd.read_csv("../data/set1_clean.tsv", sep="\t")
dev = df[~df["event"].isin(TEST_EVENTS)].reset_index(drop=True)
print("development tweets:", len(dev), " events:", dev["event"].nunique())

def make_pipe():
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("model", LogisticRegression(max_iter=1000)),
    ])

rows = []
for train_idx, val_idx in LeaveOneGroupOut().split(dev, groups=dev["event"]):
    tr, va = dev.iloc[train_idx], dev.iloc[val_idx]
    pipe = make_pipe().fit(tr["tweet_text"], tr["class_label"])   # refit from scratch each fold
    pred = pipe.predict(va["tweet_text"])
    present = sorted(va["class_label"].unique())                  # score only labels this event contains
    rows.append({"held_out_event": va["event"].iloc[0], "tweets": len(va),
                 "labels": len(present),
                 "macro_f1": f1_score(va["class_label"], pred, labels=present, average="macro"),
                 "weighted_f1": f1_score(va["class_label"], pred, average="weighted"),
                 "accuracy": accuracy_score(va["class_label"], pred)})

res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
print(res.round(3).to_string(index=False))
print("\nmean ± SD over 11 unseen events:")
for m in ["macro_f1", "weighted_f1", "accuracy"]:
    print(f"  {m:12s} {res[m].mean():.3f} ± {res[m].std():.3f}")
res.to_csv("../results/loeo_logreg.csv", index=False)

# Step 4.4: look inside the model - top words per label (trained on all 11 dev events).
pipe = make_pipe().fit(dev["tweet_text"], dev["class_label"])
words = pipe["tfidf"].get_feature_names_out()
coef = pipe["model"].coef_
print("\ntop 15 words per label:")
for i, label in enumerate(pipe["model"].classes_):
    print(f"  {label}:\n     {', '.join(words[coef[i].argsort()[-15:][::-1]])}")
