"""Step 5: how much of each difference is just noise? (bootstrap confidence intervals)

A test set is one random sample of posts. If we had drawn a different sample, every
score would come out a bit different. The bootstrap estimates that: resample the
test posts with replacement 1,000 times, recompute the score each time, and report the
middle 95% of the results as a confidence interval (CI).

If a 95% CI for a *difference* (e.g. English F1 - Hinglish F1) does not include 0,
the difference is unlikely to be noise.

Output: results/bootstrap.md
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from common import LANGS, RESULTS

N_BOOT = 1000
rng = np.random.default_rng(0)
NAMES = {"tfidf_lr": "TF-IDF + LR", "xlmr": "XLM-R", "muril": "MuRIL", "claude": "Claude (zero-shot)"}


def f1(y, p):
    return f1_score(y, p, average="macro")


def boot(parts, stat):
    """parts: {lang: (labels, {model: preds})}. Resample each language separately."""
    vals = []
    for _ in range(N_BOOT):
        sample = {}
        for lang, (y, preds) in parts.items():
            idx = rng.integers(0, len(y), len(y))
            sample[lang] = (y[idx], {m: p[idx] for m, p in preds.items()})
        vals.append(stat(sample))
    return np.percentile(vals, [2.5, 97.5])


def fmt(point, ci):
    return f"{point:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def report(models, ids=None, title=""):
    preds = {m: pd.read_csv(RESULTS / f"preds_{m}.csv").set_index("id") for m in models}
    base = preds[models[0]]
    if ids is not None:
        base = base.loc[sorted(ids)]
    parts = {}
    for lang in LANGS:
        part = base[base["lang"] == lang]
        parts[lang] = (part["label"].to_numpy(),
                       {m: preds[m].loc[part.index, "pred"].to_numpy() for m in models})

    lines = [f"### {title}", "", "Macro-F1 with 95% bootstrap CI.", "",
             "| Model | English | Hindi | Hinglish | Gap En-Hindi | Gap En-Hinglish |", "|---|---|---|---|---|---|"]
    for m in models:
        cells = []
        for lang in LANGS:
            y, p = parts[lang]
            point = f1(y, p[m])
            ci = boot({lang: parts[lang]}, lambda s, lang=lang, m=m: f1(s[lang][0], s[lang][1][m]))
            cells.append(fmt(point, ci))
        for other in ["hindi", "hinglish"]:
            gap = lambda s, o=other, m=m: f1(s["english"][0], s["english"][1][m]) - f1(s[o][0], s[o][1][m])
            sub = {"english": parts["english"], other: parts[other]}
            cells.append(fmt(gap({k: (v[0], v[1]) for k, v in sub.items()}), boot(sub, gap)))
        lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")

    if "muril" in models and "xlmr" in models:
        lines += ["", "MuRIL minus XLM-R (paired, same resampled posts):", ""]
        for lang in LANGS:
            diff = lambda s, lang=lang: f1(s[lang][0], s[lang][1]["muril"]) - f1(s[lang][0], s[lang][1]["xlmr"])
            lines.append(f"- {lang}: {fmt(diff(parts), boot({lang: parts[lang]}, diff))}")
    return "\n".join(lines) + "\n"


out = [report(["tfidf_lr", "xlmr", "muril"], title="Full test set (8,852 posts)")]
if (RESULTS / "preds_claude.csv").exists():
    ids = set(pd.read_csv(RESULTS / "preds_claude.csv")["id"])
    out.append(report(["tfidf_lr", "xlmr", "muril", "claude"], ids=ids, title="300-post sample (100 per language)"))

text = "\n".join(out)
(RESULTS / "bootstrap.md").write_text(text, encoding="utf-8")
print(text)
