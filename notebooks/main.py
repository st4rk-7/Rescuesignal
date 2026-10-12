from pathlib import Path
import pandas as pd

files = sorted(f for f in Path("../data").rglob("*.tsv") if not f.name.startswith("._"))

print(len(files), "files")
for f in files:
    print(f.parent.name, "/", f.name)



pd.set_option("display.max_colwidth", None)   # show full tweet text, don't cut it with "..."

path = Path("../data/events_set1/canada_wildfires_2016/canada_wildfires_2016_train.tsv")
one = pd.read_csv(path, sep="\t")             # sep="\t" = columns split by tabs

print(one.shape)          # (rows, columns)
print(one.columns.tolist())  # column names
print(one.head(10))       # first 10 rows
for text, label in zip(one["tweet_text"].head(10), one["class_label"].head(10)):
    print(f"[{label}]  {text}\n")
1