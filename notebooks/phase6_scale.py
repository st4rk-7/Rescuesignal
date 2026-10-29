# Step 6.2: why the 25 features need scaling, and why MinMaxScaler (not StandardScaler) for Naive Bayes.
# One example fold: train on 10 dev events, validate on maryland_floods_2018.
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import MultinomialNB
from phase6 import feature_matrix

TEST_EVENTS = ["cyclone_idai_2019", "kaikoura_earthquake_2016"]   # locked in step 4.1
df = pd.read_csv("../data/set1_clean.tsv", sep="\t")
dev = df[~df["event"].isin(TEST_EVENTS)]
tr, va = dev[dev["event"] != "maryland_floods_2018"], dev[dev["event"] == "maryland_floods_2018"]
X_tr, X_va = feature_matrix(tr["tweet_text"]), feature_matrix(va["tweet_text"])

cols = ["chars", "words", "upper_ratio", "request_w"]
print("1) RAW ranges on training events:")
print(X_tr[cols].agg(["min", "max"]).round(2).to_string())

mm = MinMaxScaler().fit(X_tr)                     # learns min/max from TRAINING events only
print("\n2) After MinMaxScaler (training events):")
print(pd.DataFrame(mm.transform(X_tr), columns=X_tr.columns)[cols].agg(["min", "max"]).round(2).to_string())

ss = StandardScaler().fit(X_tr)
std = pd.DataFrame(ss.transform(X_tr), columns=X_tr.columns)
print("\n3) After StandardScaler (training events):")
print(std[cols].agg(["min", "max"]).round(2).to_string())
print("share of values that are negative:", round((std < 0).to_numpy().mean(), 2))

print("\n4) MultinomialNB on each version:")
for name, scaler in [("MinMaxScaler", mm), ("StandardScaler", ss)]:
    try:
        nb = MultinomialNB().fit(scaler.transform(X_tr), tr["class_label"])
        print(f"   {name}: trained OK, maryland accuracy {nb.score(scaler.transform(X_va), va['class_label']):.3f}")
    except ValueError as e:
        print(f"   {name}: FAILED -> {e}")
