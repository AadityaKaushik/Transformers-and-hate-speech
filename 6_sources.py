"""Step 6: where do the labels come from?

PRISM merges several older datasets. Its full export (data/prism_cleaned.csv) has a
`source` column, which shows that each language comes from a different source.

For Hindi we also check the labels against the original annotation. Many PRISM Hindi
posts also appear in the CONSTRAINT 2021 Hindi hostility dataset (labels: non-hostile,
fake, hate, offensive, defamation). We match test posts to it and measure agreement
with Cohen's kappa: 1 = perfect agreement, 0 = no better than chance.

Needs: data/prism_cleaned.csv and data/hindi_hostility.parquet (see README), pyarrow.
Output: results/sources.md, results/hindi_constraint_match.csv
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from common import DATA, RESULTS, load_constraint, load_split, match_key

out = []

prism = pd.read_csv(DATA / "prism_cleaned.csv")
counts = pd.crosstab([prism["language"], prism["source"]], prism["hate_label"])
out.append("### PRISM sources by language (full export, before dedup)\n\n" + counts.to_markdown() + "\n")


con = load_constraint()

test = load_split("test")
test["id"] = test.index
hindi = test[test["lang"] == "hindi"].copy()
hindi["key"] = hindi["text"].map(match_key)
matched = hindi.merge(con[["key", "labels", "hostile", "hate_or_offensive"]], on="key")
matched[["id", "labels", "label"]].to_csv(RESULTS / "hindi_constraint_match.csv", index=False)

lines = [f"### Hindi test posts found in CONSTRAINT 2021: {len(matched)} of {len(hindi)}", "",
         f"- Cohen's kappa, PRISM label vs CONSTRAINT hostile: {cohen_kappa_score(matched['label'], matched['hostile']):.3f}",
         f"- Cohen's kappa, PRISM label vs CONSTRAINT hate-or-offensive: "
         f"{cohen_kappa_score(matched['label'], matched['hate_or_offensive']):.3f}", "",
         "| Model | Hindi, CONSTRAINT-derived | Hindi, rest | English | Hinglish |", "|---|---|---|---|---|"]
ids = set(matched["id"])
for name in ["tfidf_lr", "xlmr", "muril"]:
    p = pd.read_csv(RESULTS / f"preds_{name}.csv")
    h = p[p["lang"] == "hindi"]
    a, b = h[h["id"].isin(ids)], h[~h["id"].isin(ids)]
    e, g = p[p["lang"] == "english"], p[p["lang"] == "hinglish"]
    f = lambda d: f"{f1_score(d['label'], d['pred'], average='macro'):.3f}"
    lines.append(f"| {name} | {f(a)} | {f(b)} | {f(e)} | {f(g)} |")
out.append("\n".join(lines) + "\n")

text = "\n".join(out)
(RESULTS / "sources.md").write_text(text, encoding="utf-8")
print(text)
