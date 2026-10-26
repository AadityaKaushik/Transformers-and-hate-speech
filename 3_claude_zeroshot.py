"""Model 3 (used in the paper): Claude, zero-shot.

Same 300-post sample (100 per language) and the same PROMPT as 3_gemini_zeroshot.py.
Gemini's free tier kept failing with 429 (rate limit), so the sample was instead
labelled by Claude Opus 5.5 (claude-opus-5-5) in a fresh agent session. The
annotator saw only {id, text} for each post plus the prompt; gold labels were withheld.

Its raw answers are stored in results/claude_labels.jsonl ({"id": ..., "pred": 0/1}).
This script joins them with the gold labels and scores them.
"""
import json

from common import RESULTS, print_scores, save_predictions, score_by_language

sample_ids = __import__("3_gemini_zeroshot").sample_ids

sample = sample_ids()
labels = {}
for line in (RESULTS / "claude_labels.jsonl").read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    labels[row["id"]] = row["pred"]

missing = set(sample["id"]) - set(labels)
assert not missing, f"{len(missing)} sample posts have no Claude label"

sample["pred"] = sample["id"].map(labels).astype(int)
print_scores("claude", score_by_language(sample))
save_predictions(sample, "claude")
