# Step 6.1: the 25 hand-made features from proposal Table 3, computed per tweet.
import re
import string
import pandas as pd
from phase2 import clean

# Lexicons from proposal Table 3 (rows 17-25), fixed in advance: never edited after looking at scores.
LEXICONS = {
    "request_w":    {"need", "needs", "needed", "please", "send", "require", "shortage", "lack", "help"},
    "urgency_w":    {"urgent", "emergency", "asap", "immediately", "critical", "trapped", "sos"},
    "donation_w":   {"donate", "donation", "volunteer", "fundraiser", "relief", "supplies", "contribute"},
    "infra_w":      {"bridge", "road", "power", "electricity", "outage", "collapsed", "damaged", "destroyed", "building"},
    "casualty_w":   {"dead", "killed", "death", "toll", "injured", "wounded", "hospital", "casualties"},
    "displace_w":   {"evacuate", "evacuation", "shelter", "displaced", "homeless", "flee", "refugees"},
    "missing_w":    {"missing", "found", "search", "whereabouts", "unaccounted", "located"},
    "sympathy_w":   {"pray", "prayers", "thoughts", "condolences", "heartbroken", "rip"},
    "caution_w":    {"warning", "alert", "advisory", "avoid", "stay", "safety", "prepare"},
}
# clean() turns "don't"/"can't" into "... not", so contractions are counted as "not".
NEGATIONS = {"not", "no", "never", "none", "nothing", "cannot", "without"}
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

def features(text):
    raw = str(text)                       # statistical + social-media features use the original text
    tokens = raw.split()
    letters = sum(c.isalpha() for c in raw)
    digits = sum(c.isdigit() for c in raw)
    upper = sum("A" <= c <= "Z" for c in raw)
    words = clean(raw).split()            # negation + lexicon counts use the normalised text
    f = {"words":        len(tokens),                                            # 1-4 length
         "chars":        len(raw),
         "avg_word_len": sum(map(len, tokens)) / len(tokens) if tokens else 0,
         "unique_ratio": len(set(tokens)) / len(tokens) if tokens else 0,
         "digits":       digits,                                                 # 5-6 digits
         "digit_ratio":  digits / len(raw) if raw else 0,
         "upper":        upper,                                                  # 7-8 uppercase
         "upper_ratio":  upper / letters if letters else 0,
         "punct":        sum(c in string.punctuation for c in raw),              # 9-11 punctuation
         "exclaims":     raw.count("!"),
         "questions":    raw.count("?"),
         "hashtags":     len(re.findall(r"#\w+", raw)),                          # 12-15 social media
         "mentions":     len(re.findall(r"@\w+", raw)),
         "urls":         len(re.findall(r"https?://\S+", raw)),
         "emoji":        len(EMOJI.findall(raw)),
         "negations":    sum(w in NEGATIONS for w in words)}                     # 16
    f.update({name: sum(w in lex for w in words) for name, lex in LEXICONS.items()})   # 17-25
    return f

def feature_matrix(texts):
    """Tweets -> table of 25 numbers per tweet (used inside a Pipeline via FunctionTransformer)."""
    return pd.DataFrame([features(t) for t in texts])

if __name__ == "__main__":
    f = features("Need HELP!! 3 dead? We can't stay #Harvey @fema http://t.co/x")
    assert len(f) == 25
    assert (f["request_w"], f["casualty_w"], f["caution_w"], f["negations"]) == (2, 1, 1, 1)
    assert (f["hashtags"], f["mentions"], f["urls"], f["exclaims"], f["questions"], f["digits"]) == (1, 1, 1, 2, 1, 1)

    TEST_EVENTS = ["cyclone_idai_2019", "kaikoura_earthquake_2016"]   # locked in step 4.1
    df = pd.read_csv("../data/set1_clean.tsv", sep="\t")
    dev = df[~df["event"].isin(TEST_EVENTS)]
    X = pd.DataFrame([features(t) for t in dev["tweet_text"]], index=dev.index)
    means = X.groupby(dev["class_label"]).mean()
    pd.set_option("display.width", 250)
    print("average per tweet, by label (dev events only):")
    print(means.T.round(2).to_string())
    # how much a feature differs between labels: highest label mean / overall mean
    print("\nseparation (max label mean / overall mean), top first:")
    print((means.max() / X.mean()).sort_values(ascending=False).round(1).to_string())
