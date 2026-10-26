"""Model 3: Gemini, zero-shot (the "smart student with no preparation").

Zero-shot = no training at all. We just describe the task in a prompt and ask
the LLM to label each post. Calling an API for all 8,852 test posts would be
slow, so we label a fixed sample: 100 posts per language = 300 posts.
4_analysis.py scores the other models on the same 300 posts for a fair comparison.

Answers are cached in results/gemini_cache.jsonl, so if the script stops
(rate limit, no internet) just run it again and it continues where it left off.

Usage:
    set GEMINI_API_KEY=...        (PowerShell: $env:GEMINI_API_KEY="...")
    python 3_gemini_zeroshot.py [--model gemini-2.5-flash]
"""
import argparse
import json
import os
import time

import pandas as pd
from google import genai
from google.genai import types

from common import LANGS, RESULTS, load_split, print_scores, save_predictions, score_by_language

PER_LANG = 100

PROMPT = """You are annotating social media posts for a hate speech dataset.
Posts may be in English, Hindi (Devanagari), or Hinglish (Hindi written in Latin script, mixed with English).

Label 1 (hate) if the post attacks, insults, threatens, or demeans a person or group,
or uses abusive / profane language against someone.
Label 0 (not hate) otherwise, including posts that merely discuss or report on hateful events.

Post: {text}

Reply with JSON only: {{"label": 0 or 1}}"""


def sample_ids():
    test = load_split("test")
    test["id"] = test.index
    return pd.concat(
        [test[test["lang"] == lang].sample(PER_LANG, random_state=42) for lang in LANGS]
    )


def ask(client, model, text):
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=PROMPT.format(text=text),
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    # Turn off safety blocking: we *need* the model to read hateful text to label it.
                    safety_settings=[
                        types.SafetySetting(category=c, threshold="BLOCK_NONE")
                        for c in ["HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_HARASSMENT",
                                  "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
                    ],
                ),
            )
            return int(json.loads(resp.text)["label"])
        except Exception as e:  # rate limits, blocked responses, bad JSON
            wait = 2 ** attempt * 5
            print(f"  retry {attempt + 1} in {wait}s: {str(e)[:100]}")
            time.sleep(wait)
    return None  # gave up; counted as "no answer" below


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemini-3.8-flash")
    args = parser.parse_args()

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    sample = sample_ids()

    RESULTS.mkdir(exist_ok=True)
    cache_path = RESULTS / "gemini_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[row["id"]] = row["pred"]

    with open(cache_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(sample.itertuples()):
            if row.id in cache:
                continue
            pred = ask(client, args.model, row.text)
            cache[row.id] = pred
            f.write(json.dumps({"id": row.id, "pred": pred}) + "\n")
            f.flush()
            print(f"{i + 1}/{len(sample)} [{row.lang}] gold={row.label} gemini={pred}")

    sample["pred"] = sample["id"].map(cache)
    missing = sample["pred"].isna().sum()
    if missing:
        print(f"{missing} posts got no answer; scored as 0 (not hate)")
    sample["pred"] = sample["pred"].fillna(0).astype(int)

    print_scores("gemini", score_by_language(sample))
    save_predictions(sample, "gemini")


if __name__ == "__main__":
    main()
