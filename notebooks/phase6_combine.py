# Step 6.3: put TF-IDF and the 25 scaled features side by side with a ColumnTransformer.
# make_pipeline() is reused by the ablation in step 6.4.
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.svm import LinearSVC
from phase2 import clean
from phase6 import feature_matrix

def make_pipeline(model, use_tfidf=True, use_feats=True):
    """(A) TF-IDF only, (B) features only, (C) both. Every learned step is refitted inside each fold."""
    parts = []
    if use_tfidf:
        parts.append(("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                      "tweet_text"))
    if use_feats:
        parts.append(("feats", Pipeline([("count", FunctionTransformer(feature_matrix)), ("scale", MinMaxScaler())]),
                      "tweet_text"))
    return Pipeline([("columns", ColumnTransformer(parts)), ("model", model)])

if __name__ == "__main__":
    TEST_EVENTS = ["cyclone_idai_2019", "kaikoura_earthquake_2016"]   # locked in step 4.1
    df = pd.read_csv("../data/set1_clean.tsv", sep="\t")
    dev = df[~df["event"].isin(TEST_EVENTS)]
    tr = dev[dev["event"] != "maryland_floods_2018"]

    for name, t, f in [("A TF-IDF only", True, False), ("B features only", False, True), ("C both", True, True)]:
        pipe = make_pipeline(LinearSVC(C=0.1, class_weight="balanced"), t, f).fit(tr[["tweet_text"]], tr["class_label"])
        cols = pipe.named_steps["columns"].transform(tr[["tweet_text"]].head(1)).shape[1]
        print(f"{name:16} -> {cols:,} columns per tweet")
