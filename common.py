"""Shared helpers: loading the data and scoring predictions."""
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LANGS = ["english", "hindi", "hinglish"]


def load_split(name):
    """Load train / val / test. Each row: text, label (0 = not hate, 1 = hate), lang."""
    df = pd.read_csv(DATA / f"{name}.csv")
    df["text"] = df["text"].astype(str)
    return df


def match_key(text):
    # Match on the first 60 letters/digits only. PRISM and CONSTRAINT clean text
    # differently (URLs, mentions, punctuation, spacing), so exact matching fails.
    text = re.sub(r"http\S+|@\w+", "", str(text).lower())
    return "".join(ch for ch in text if ch.isalnum())[:60]


def load_constraint():
    """CONSTRAINT 2021 Hindi hostility data, one row per unambiguous match key."""
    con = pd.read_parquet(DATA / "hindi_hostility.parquet")
    con.columns = ["uid", "post", "labels"]
    con["key"] = con["post"].map(match_key)
    con = con.drop_duplicates("key", keep=False)  # drop keys that would match ambiguously
    con["hostile"] = (con["labels"] != "non-hostile").astype(int)
    con["hate_or_offensive"] = con["labels"].str.contains("hate|offensive").astype(int)
    return con


def macro_f1(y_true, y_pred):
    # Macro-F1 = average of the F1 for class 0 and class 1, so a model can't
    # look good by only getting the bigger class right. HASOC uses this metric too.
    return f1_score(y_true, y_pred, average="macro")


def score_by_language(df):
    """df needs columns label, pred, lang. Returns {'all': f1, 'english': f1, ...}."""
    scores = {"all": macro_f1(df["label"], df["pred"])}
    for lang in LANGS:
        part = df[df["lang"] == lang]
        scores[lang] = macro_f1(part["label"], part["pred"])
    return scores


def print_scores(name, scores):
    row = "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
    print(f"{name:<12} {row}")


def save_predictions(df, model_name):
    """Every model writes the same format, so 4_analysis.py can compare them."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"preds_{model_name}.csv"
    df[["id", "text", "lang", "label", "pred"]].to_csv(path, index=False)
    print(f"saved {path}")
