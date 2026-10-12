# Step 2.1: clean(text) - turn a messy tweet into normalised text for TF-IDF.
import re
import html
import pandas as pd

def clean(text):
    t = html.unescape(str(text)).lower()           # "&amp;" -> "&", then Water = water
    t = re.sub(r"^rt\s+", "", t)                    # drop retweet marker
    t = re.sub(r"https?://\S+", " URL ", t)         # links -> URL
    t = re.sub(r"@\w+", " USER ", t)                # @names -> USER
    t = t.replace("#", " ")                         # #floodrelief -> floodrelief (keep the word)
    t = re.sub(r"\d+", " NUM ", t)                  # 30, 2017 -> NUM
    t = re.sub(r"n't\b", " not", t)                 # don't -> do not (keep negation)
    t = re.sub(r"[^a-zA-Z ]", " ", t)               # drop other symbols / emoji
    return " ".join(t.split())                      # collapse spaces

if __name__ == "__main__":
    df = pd.read_csv("../data/set1_clean.tsv", sep="\t")
    for text in df["tweet_text"].sample(5, random_state=1):
        print("BEFORE:", text)
        print("AFTER: ", clean(text), "\n")
    print("TEST:  ", clean("We do NOT need food!! @RedCross 30 people trapped #HarveyRelief http://t.co/x"))
