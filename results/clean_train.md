### Retraining without CONSTRAINT-matched Hindi posts
Removed 1114 train and 195 val posts. Test set unchanged.

| Model | Training data | English | Hindi (all) | Hindi, matched | Hindi, rest | Hinglish |
|---|---|---|---|---|---|---|
| tfidf_lr | full | 0.777 | 0.629 | 0.519 | 0.650 | 0.623 |
| tfidf_lr | clean | 0.775 | 0.616 | 0.471 | 0.643 | 0.624 |
| xlmr | full | 0.816 | 0.615 | 0.405 | 0.641 | 0.666 |
| xlmr | clean | 0.809 | 0.624 | 0.398 | 0.656 | 0.647 |
| muril | full | 0.802 | 0.628 | 0.396 | 0.660 | 0.669 |
| muril | clean | 0.800 | 0.632 | 0.435 | 0.662 | 0.671 |

Change on Hindi, rest (clean - full), paired bootstrap 95% CI:

- tfidf_lr: -0.006 [-0.017, +0.004]
- xlmr: +0.016 [+0.002, +0.031]
- muril: +0.002 [-0.011, +0.014]
