### PRISM sources by language (full export, before dedup)

|                                 |    0 |    1 |
|:--------------------------------|-----:|-----:|
| ('english', 'english_dataset')  | 7500 | 7500 |
| ('hindi', 'hindi_dataset')      | 5411 | 4356 |
| ('hinglish', 'hate_speech_tsv') | 2914 | 1660 |
| ('hinglish', 'profanity_list')  |    0 |  209 |

### Hindi test posts found in CONSTRAINT 2021: 557 of 2900

- Cohen's kappa, PRISM label vs CONSTRAINT hostile: -0.006
- Cohen's kappa, PRISM label vs CONSTRAINT hate-or-offensive: -0.025

| Model | Hindi, CONSTRAINT-derived | Hindi, rest | English | Hinglish |
|---|---|---|---|---|
| tfidf_lr | 0.519 | 0.650 | 0.777 | 0.623 |
| xlmr | 0.405 | 0.641 | 0.816 | 0.666 |
| muril | 0.396 | 0.660 | 0.802 | 0.669 |
