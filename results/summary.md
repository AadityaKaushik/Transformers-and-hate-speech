### Macro-F1 on the full test set (8,852 posts)

| Model | All | English | Hindi | Hinglish | Gap (English - Hinglish) |
|---|---|---|---|---|---|
| TF-IDF + LR | 0.711 | 0.777 | 0.629 | 0.623 | +0.154 |
| XLM-R | 0.731 | 0.816 | 0.615 | 0.666 | +0.150 |
| MuRIL | 0.731 | 0.802 | 0.628 | 0.669 | +0.132 |

### Macro-F1 on the 300-post sample (100 per language), all models

| Model | All | English | Hindi | Hinglish | Gap (English - Hinglish) |
|---|---|---|---|---|---|
| TF-IDF + LR | 0.676 | 0.786 | 0.662 | 0.571 | +0.215 |
| XLM-R | 0.692 | 0.867 | 0.507 | 0.700 | +0.167 |
| MuRIL | 0.700 | 0.817 | 0.628 | 0.648 | +0.169 |
| Claude (zero-shot) | 0.622 | 0.651 | 0.608 | 0.605 | +0.046 |

### Claude (zero-shot) disagrees with the gold label

- english: 34%
- hindi: 39%
- hinglish: 39%
