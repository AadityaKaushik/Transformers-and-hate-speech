"""Step 4: compare every model, per language, and pull out mistakes to read by hand.

Reads every results/preds_*.csv that exists, so you can run it after any step.

Outputs:
  results/summary.md            the results tables (copy into README)
  results/f1_by_language.png    bar chart of the same numbers
  results/errors_to_review.csv  mistakes to re-label by hand (see README)
"""
import matplotlib.pyplot as plt
import pandas as pd

from common import LANGS, RESULTS, score_by_language

# Fixed display order + fixed colours, so a model keeps its colour in every chart.
MODELS = {
    "tfidf_lr": ("TF-IDF + LR", "#2a78d6"),
    "xlmr": ("XLM-R", "#eb6834"),
    "muril": ("MuRIL", "#1baf7a"),
    "claude": ("Claude (zero-shot)", "#eda100"),
    "gemini": ("Gemini (zero-shot)", "#e87ba4"),
}

preds = {name: pd.read_csv(RESULTS / f"preds_{name}.csv")
         for name in MODELS if (RESULTS / f"preds_{name}.csv").exists()}
if not preds:
    raise SystemExit("No predictions yet. Run 1_baseline.py first.")


def table(rows, title):
    lines = [f"### {title}", "", "| Model | All | English | Hindi | Hinglish | Gap (English - Hinglish) |",
             "|---|---|---|---|---|---|"]
    for name, s in rows.items():
        lines.append(f"| {MODELS[name][0]} | {s['all']:.3f} | {s['english']:.3f} | {s['hindi']:.3f} "
                     f"| {s['hinglish']:.3f} | {s['english'] - s['hinglish']:+.3f} |")
    return "\n".join(lines) + "\n"


# Zero-shot LLMs only labelled a 300-post sample, so they get their own table.
ZERO_SHOT = [n for n in ("claude", "gemini") if n in preds]

# Table 1: full test set, trained models only.
full = {n: score_by_language(df) for n, df in preds.items() if n not in ZERO_SHOT}
out = [table(full, "Macro-F1 on the full test set (8,852 posts)")]

# Table 2: everyone on the same 300-post sample the LLM labelled.
same = None
if ZERO_SHOT:
    ids = set(preds[ZERO_SHOT[0]]["id"])
    same = {n: score_by_language(df[df["id"].isin(ids)]) for n, df in preds.items()}
    out.append(table(same, "Macro-F1 on the 300-post sample (100 per language), all models"))

    # How often does the LLM disagree with the dataset label, per language?
    for n in ZERO_SHOT:
        z = preds[n]
        disagree = (z["pred"] != z["label"]).groupby(z["lang"]).mean()
        out.append(f"### {MODELS[n][0]} disagrees with the gold label\n\n" + "\n".join(
            f"- {lang}: {disagree[lang]:.0%}" for lang in LANGS) + "\n")

summary = "\n".join(out)
(RESULTS / "summary.md").write_text(summary, encoding="utf-8")
print(summary)


def bar_chart(scores_by_model, subtitle, filename):
    """Grouped bars: one group per language, one bar per model."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    groups = ["all"] + LANGS
    width = 0.8 / len(scores_by_model)
    for i, (name, scores) in enumerate(scores_by_model.items()):
        label, colour = MODELS[name]
        xs = [g + i * width for g in range(len(groups))]
        bars = ax.bar(xs, [scores[g] for g in groups], width - 0.02, color=colour, label=label)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color="#52514e", padding=2)
    ax.set_xticks([g + width * (len(scores_by_model) - 1) / 2 for g in range(len(groups))])
    ax.set_xticklabels(["All", "English", "Hindi", "Hinglish"], color="#0b0b0b")
    ax.set_ylabel("Macro-F1", color="#52514e")
    ax.set_ylim(0, 1)
    ax.set_title(f"Hate speech detection: macro-F1 by language ({subtitle})", color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="#52514e", length=0)
    ax.legend(frameon=False, ncol=len(scores_by_model), loc="upper left", bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    fig.savefig(RESULTS / filename, dpi=150)
    print(f"saved {RESULTS / filename}")


bar_chart(full, "full test set", "f1_by_language.png")
if same:
    bar_chart(same, "300-post sample", "f1_by_language_sample.png")

# Error analysis: 60 mistakes of the best fine-tuned model (20 per language), which
# I re-labelled by hand (my_label: 1 = hate, 0 = not hate) without changing the gold labels.
# Each mistake means model != gold, so my label sides with one of them:
#   my_label == gold  -> a real model error
#   my_label == model -> the gold label is probably wrong
# The sheet is only created once, so hand labels are never overwritten.
review_path = RESULTS / "errors_to_review.csv"
if review_path.exists():
    rev = pd.read_csv(review_path).dropna(subset=["my_label"])
    if len(rev):
        rev["gold_looks_wrong"] = rev["my_label"] != rev["label"]
        print(f"\nHand re-labelling of {len(rev)} model mistakes:")
        print("share where my label disagrees with gold (= gold label looks wrong):")
        print(rev.groupby("lang")["gold_looks_wrong"].agg(["mean", "sum", "count"]).round(2))
        print(f"overall: {rev['gold_looks_wrong'].mean():.0%}")
        trained = [c for c in rev.columns if c.endswith("_pred") and not c.startswith(("claude", "gemini"))]
        all_wrong = (rev[trained].ne(rev["label"], axis=0)).all(axis=1)
        print(f"when all {len(trained)} trained models contradict gold: {rev[all_wrong]['gold_looks_wrong'].mean():.0%} "
              f"(n={all_wrong.sum()}); otherwise: {rev[~all_wrong]['gold_looks_wrong'].mean():.0%} (n={(~all_wrong).sum()})")
else:
    best = max(full, key=lambda n: full[n]["all"])
    df = preds[best]
    errors = df[df["pred"] != df["label"]]
    sample = pd.concat([errors[errors["lang"] == lang].sample(min(20, (errors["lang"] == lang).sum()),
                                                              random_state=0) for lang in LANGS])
    for other, odf in preds.items():
        if other != best:
            sample[f"{other}_pred"] = sample["id"].map(odf.set_index("id")["pred"])
    sample = sample.rename(columns={"pred": f"{best}_pred"})
    sample["my_label"] = ""
    sample["notes"] = ""
    sample.to_csv(review_path, index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows Hindi
    print(f"\nWrote {len(sample)} {best} mistakes to {review_path}")
    print("Read each post and fill my_label with your own judgement: 1 = hate, 0 = not hate.")
