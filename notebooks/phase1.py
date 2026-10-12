# Step 1.3: put all events in one table, with an `event` column.
from pathlib import Path
import pandas as pd

DATA = Path("../data/events_set1")
parts = []
for f in sorted(DATA.rglob("*.tsv")):
    if f.name.startswith("._"):          # skip macOS junk files
        continue
    part = pd.read_csv(f, sep="\t")
    part["event"] = f.parent.name         # folder name = event name
    parts.append(part)

df = pd.concat(parts, ignore_index=True)  # stack all 39 files into one table
print(df.shape)
print(df["event"].value_counts())         # tweets per event

# Step 1.4: count tweets per label and per event, draw a bar chart of each.
import matplotlib
matplotlib.use("Agg")                     # save to file, no window
import matplotlib.pyplot as plt

OUT = Path("../results/figures")
OUT.mkdir(parents=True, exist_ok=True)

def bar_chart(counts, title, filename):
    counts = counts.sort_values()         # smallest at bottom, biggest at top
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(counts) + 1))
    ax.set_facecolor("#fcfcfb")
    ax.barh(counts.index, counts.values, color="#2a78d6", height=0.7)
    for y, v in enumerate(counts.values):  # write the number next to each bar
        ax.text(v, y, f" {v:,}", va="center", fontsize=9, color="#444")
    ax.set_title(title, loc="left")
    ax.set_xlabel("Number of tweets")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

label_counts = df["class_label"].value_counts()
print(label_counts)
print("biggest / smallest =", round(label_counts.max() / label_counts.min(), 1))
bar_chart(label_counts, "Tweets per label (all 13 events)", "tweets_per_label.png")
bar_chart(df["event"].value_counts(), "Tweets per event", "tweets_per_event.png")

# Step 1.6: audit - empty texts, duplicates, texts shared across events.
import re

def normalise(text):
    t = str(text).lower()
    t = re.sub(r"^rt\s+", "", t)              # drop retweet marker
    t = re.sub(r"https?://\S+", "", t)        # drop links
    t = re.sub(r"@\w+", "", t)                # drop @names
    t = re.sub(r"[^a-z0-9 ]", " ", t)         # drop punctuation / emoji
    return " ".join(t.split())                # collapse spaces

df["norm"] = df["tweet_text"].map(normalise)

print("\n--- AUDIT ---")
print("empty texts:          ", (df["norm"] == "").sum())
print("duplicate tweet_ids:  ", df["tweet_id"].duplicated().sum())
print("exact duplicate texts:", df["tweet_text"].duplicated().sum())
print("normalised duplicates:", df["norm"].duplicated().sum())

groups = df[df["norm"].duplicated(keep=False)].groupby("norm")
cross_event = groups["event"].nunique().gt(1)
cross_label = groups["class_label"].nunique().gt(1)
print("dup groups:           ", len(groups))
print("  in >1 event:        ", cross_event.sum(),
      "groups,", df["norm"].isin(cross_event[cross_event].index).sum(), "tweets")
print("  with >1 label:      ", cross_label.sum(), "groups")

print("\nexamples of texts in >1 event:")
for norm in cross_event[cross_event].index[:3]:
    rows = df[df["norm"] == norm]
    print(" ", rows["tweet_text"].iloc[0][:90])
    print("   events:", sorted(rows["event"].unique()), "labels:", sorted(rows["class_label"].unique()))

# Decision (logbook): drop the extra copies of duplicate texts, keep the first one.
before = len(df)
df = df.drop_duplicates(subset="norm", keep="first").drop(columns="norm")
print(f"\nremoved {before - len(df)} duplicates -> {len(df)} tweets")
df.to_csv("../data/set1_clean.tsv", sep="\t", index=False)   # input for Phase 2+
