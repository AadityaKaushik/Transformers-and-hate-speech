"""Step 8: retrain without the broken Hindi labels.

Step 6 showed that PRISM Hindi posts which also appear in CONSTRAINT 2021 have labels
unrelated to the original annotation. Here we remove those posts from train and val,
retrain, and ask: does the Hindi score go up on the posts whose labels we trust?

The test set is NOT changed, so the numbers stay comparable with steps 1-6. The key
column is "Hindi, rest": test posts outside the CONSTRAINT match.

First run: writes data/train_clean.csv and data/val_clean.csv and retrains TF-IDF.
Upload both files to 2b_finetune_clean_colab.ipynb, put the downloaded
preds_muril_clean.csv / preds_xlmr_clean.csv in results/, then run this again.

Needs: data/hindi_hostility.parquet (see README), pyarrow.
Output: results/clean_train.md
"""
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import DATA, RESULTS, load_constraint, load_split, macro_f1, save_predictions, match_key

con_keys = set(load_constraint()["key"])


def drop_matched(df):
    hit = (df["lang"] == "hindi") & df["text"].map(match_key).isin(con_keys)
    return df[~hit], int(hit.sum())


train, val, test = load_split("train"), load_split("val"), load_split("test")
test["id"] = test.index
train_clean, n_train = drop_matched(train)
val_clean, n_val = drop_matched(val)
train_clean.to_csv(DATA / "train_clean.csv", index=False)
val_clean.to_csv(DATA / "val_clean.csv", index=False)
print(f"removed {n_train} Hindi train posts and {n_val} Hindi val posts")

# Same TF-IDF + LR recipe as 1_baseline.py, C picked on the cleaned validation set.
word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
X_train = hstack([word_vec.fit_transform(train_clean["text"]), char_vec.fit_transform(train_clean["text"])]).tocsr()
X_val = hstack([word_vec.transform(val_clean["text"]), char_vec.transform(val_clean["text"])]).tocsr()
X_test = hstack([word_vec.transform(test["text"]), char_vec.transform(test["text"])]).tocsr()
best_c = max([0.5, 1, 2, 4, 8], key=lambda c: macro_f1(
    val_clean["label"], LogisticRegression(C=c, max_iter=2000).fit(X_train, train_clean["label"]).predict(X_val)))
clf = LogisticRegression(C=best_c, max_iter=2000).fit(X_train, train_clean["label"])
test["pred"] = clf.predict(X_test)
print(f"tfidf best C={best_c}")
save_predictions(test, "tfidf_lr_clean")

matched_ids = set(pd.read_csv(RESULTS / "hindi_constraint_match.csv")["id"])
GROUPS = {
    "English": lambda d: d["lang"] == "english",
    "Hindi (all)": lambda d: d["lang"] == "hindi",
    "Hindi, matched": lambda d: (d["lang"] == "hindi") & d["id"].isin(matched_ids),
    "Hindi, rest": lambda d: (d["lang"] == "hindi") & ~d["id"].isin(matched_ids),
    "Hinglish": lambda d: d["lang"] == "hinglish",
}


def paired_ci(full, clean, n_boot=1000, seed=0):
    """95% bootstrap interval for (clean - full) macro-F1 on the same posts."""
    rng = np.random.default_rng(seed)
    y, a, b = full["label"].to_numpy(), full["pred"].to_numpy(), clean["pred"].to_numpy()
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        diffs.append(macro_f1(y[i], b[i]) - macro_f1(y[i], a[i]))
    return np.percentile(diffs, [2.5, 97.5])


lines = [f"### Retraining without CONSTRAINT-matched Hindi posts",
         f"Removed {n_train} train and {n_val} val posts. Test set unchanged.", "",
         "| Model | Training data | " + " | ".join(GROUPS) + " |",
         "|---|---|" + "---|" * len(GROUPS)]
diff_lines = ["", "Change on Hindi, rest (clean - full), paired bootstrap 95% CI:", ""]
for name in ["tfidf_lr", "xlmr", "muril"]:
    clean_path = RESULTS / f"preds_{name}_clean.csv"
    if not clean_path.exists():
        lines.append(f"| {name} | clean | (run 2b_finetune_clean_colab.ipynb) |")
        continue
    runs = {"full": pd.read_csv(RESULTS / f"preds_{name}.csv"), "clean": pd.read_csv(clean_path)}
    for tag, p in runs.items():
        cells = [f"{macro_f1(p[g(p)]['label'], p[g(p)]['pred']):.3f}" for g in GROUPS.values()]
        lines.append(f"| {name} | {tag} | " + " | ".join(cells) + " |")
    rest = GROUPS["Hindi, rest"]
    full, clean = (runs[t][rest(runs[t])].sort_values("id") for t in ["full", "clean"])
    lo, hi = paired_ci(full, clean)
    d = macro_f1(clean["label"], clean["pred"]) - macro_f1(full["label"], full["pred"])
    diff_lines.append(f"- {name}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

text = "\n".join(lines + diff_lines) + "\n"
(RESULTS / "clean_train.md").write_text(text, encoding="utf-8")
print(text)
