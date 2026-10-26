"""Step 7: blind check of label quality.

The first hand check (errors_to_review.csv) was not blind: model predictions were
visible. Here a human labels random test posts seeing ONLY the text, and we measure
agreement with the gold label per language (Cohen's kappa).

If gold labels were equally reliable everywhere, agreement would be similar for
English, Hindi and Hinglish.

First run: creates results/blind_labels.csv (30 random posts per language, shuffled,
text only). Fill `my_label` (1 = hate, 0 = not hate) without looking anything up.
Second run: scores it. To add a second annotator, copy the file to
results/blind_labels_2.csv and have them fill it; agreement between you is reported too.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import LANGS, RESULTS, load_split

PER_LANG = 30
sheet = RESULTS / "blind_labels.csv"

test = load_split("test")
test["id"] = test.index

if not sheet.exists():
    seen = set(pd.read_csv(RESULTS / "errors_to_review.csv")["id"])
    pool = test[~test["id"].isin(seen)]
    picked = pd.concat([pool[pool["lang"] == lang].sample(PER_LANG, random_state=123) for lang in LANGS])
    picked = picked.sample(frac=1, random_state=1)  # mix languages so you can't anchor on one
    out = picked[["id", "text"]].copy()
    out["my_label"] = ""
    out.to_csv(sheet, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} posts to {sheet}. Fill my_label with 1 (hate) or 0 (not hate).")
    raise SystemExit

def kappa_ci(part, n_boot=1000, seed=0):
    """95% bootstrap interval for kappa between gold and my label."""
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        s = part.iloc[rng.integers(0, len(part), len(part))]
        if s["label"].nunique() > 1 and s["my_label"].nunique() > 1:
            ks.append(cohen_kappa_score(s["label"], s["my_label"]))
    return np.percentile(ks, [2.5, 97.5])


lab = pd.read_csv(sheet).dropna(subset=["my_label"])
lab = lab.merge(test[["id", "label", "lang"]], on="id")
muril = pd.read_csv(RESULTS / "preds_muril.csv")[["id", "pred"]]
lab = lab.merge(muril, on="id", how="left")
lines = [f"### Blind re-annotation ({len(lab)} posts)", "",
         "| Language | n | Agreement with gold | Cohen's kappa [95% CI] | MuRIL agrees with gold | Kappa me vs MuRIL |",
         "|---|---|---|---|---|---|"]
for lang in LANGS:
    part = lab[lab["lang"] == lang]
    if len(part):
        lo, hi = kappa_ci(part)
        lines.append(f"| {lang} | {len(part)} | {(part['my_label'] == part['label']).mean():.0%} "
                     f"| {cohen_kappa_score(part['label'], part['my_label']):.2f} [{lo:.2f}, {hi:.2f}] "
                     f"| {(part['pred'] == part['label']).mean():.0%} "
                     f"| {cohen_kappa_score(part['my_label'], part['pred']):.2f} |")

second = RESULTS / "blind_labels_2.csv"
if second.exists():
    two = pd.read_csv(second).dropna(subset=["my_label"]).merge(lab, on="id", suffixes=("_b", ""))
    lines += ["", f"Inter-annotator kappa (n={len(two)}): "
              f"{cohen_kappa_score(two['my_label'], two['my_label_b']):.2f}"]

text = "\n".join(lines) + "\n"
(RESULTS / "blind_check.md").write_text(text, encoding="utf-8")
print(text)
