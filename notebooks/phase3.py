# Step 3.1: random 80/20 split, stratified by label (the "cheating" reference split).
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/set1_clean.tsv", sep="\t")
train, test = train_test_split(df, test_size=0.2, stratify=df["class_label"], random_state=42)

print("train:", len(train), " test:", len(test))
share = pd.DataFrame({"train %": train["class_label"].value_counts(normalize=True) * 100,
                      "test %": test["class_label"].value_counts(normalize=True) * 100}).round(1)
print(share)
both = set(train["event"]) & set(test["event"])
print("events in BOTH train and test:", len(both), "of", df["event"].nunique())
print("harvey tweets -> train:", (train["event"] == "hurricane_harvey_2017").sum(),
      " test:", (test["event"] == "hurricane_harvey_2017").sum())

# Step 3.2: Logistic Regression on TF-IDF.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from phase2 import clean

tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = tfidf.fit_transform(train["tweet_text"].map(clean))   # learn vocabulary on TRAIN only
X_test = tfidf.transform(test["tweet_text"].map(clean))         # reuse it on test

model = LogisticRegression(max_iter=1000)
model.fit(X_train, train["class_label"])
pred = model.predict(X_test)

print("\nvocabulary size:", X_train.shape[1], " weights learned:", model.coef_.size)
for text, true, p in list(zip(test["tweet_text"], test["class_label"], pred))[:5]:
    print(f"[{'OK ' if true == p else 'BAD'}] true={true}  pred={p}\n      {text[:100]}")

words = tfidf.get_feature_names_out()
for label in ["requests_or_urgent_needs", "injured_or_dead_people"]:
    i = list(model.classes_).index(label)
    print(f"\ntop words for {label}:", list(words[model.coef_[i].argsort()[-8:][::-1]]))

# Step 3.3: scores.
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("\naccuracy:   ", round(accuracy_score(test["class_label"], pred), 3))
print("macro F1:   ", round(f1_score(test["class_label"], pred, average="macro"), 3))
print("weighted F1:", round(f1_score(test["class_label"], pred, average="weighted"), 3))
print(classification_report(test["class_label"], pred, digits=2, zero_division=0))

# Step 3.4: confusion matrix (rows = true label, columns = predicted label).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

labels = sorted(df["class_label"].unique())
cm = confusion_matrix(test["class_label"], pred, labels=labels)
short = [l.split("_")[0] for l in labels]                 # caution, displaced, ...
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm, display_labels=short).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
ax.set_title("Confusion matrix: random split, Logistic Regression", loc="left")
fig.tight_layout()
fig.savefig("../results/figures/cm_random_logreg.png", dpi=150)

pairs = [(cm[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j]
print("\nmost common mistakes (count, true -> predicted):")
for n, t, p in sorted(pairs, reverse=True)[:5]:
    print(f"  {n:4d}  {t} -> {p}")

# Step 3.5: the same thing as ONE Pipeline: clean -> TF-IDF -> model.
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("model", LogisticRegression(max_iter=1000)),
])
pipe.fit(train["tweet_text"], train["class_label"])          # every step learns from TRAIN only
pipe_pred = pipe.predict(test["tweet_text"])                  # every step only transforms TEST
print("\npipeline macro F1:", round(f1_score(test["class_label"], pipe_pred, average="macro"), 3),
      " same predictions as 3.2:", (pipe_pred == pred).all())
