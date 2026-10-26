# What you need to understand (interview prep)

Read this once tonight and once before the interview. Each section is something you could be asked.

## 1. The problem in one sentence
Automatically detecting hate speech is hard for Indian social media because people write in
Hindi, English, and Hinglish (Hindi in English letters, mixed with English), and most models
are trained mainly on clean English.

**Code-mixing / code-switching** = switching languages within one sentence ("yaar ye banda total bakwaas hai").
**Transliteration** = writing a language in another script ("bakwaas" instead of "बकवास"). There is no
standard spelling, so the same word appears as "besharam", "besaram", "beshrm", and so on.

## 2. Train / validation / test
- **Train**: the model learns from these.
- **Validation**: used to choose settings (e.g. C for logistic regression, the number of epochs).
- **Test**: used **once**, at the end, to report results. If you tune on test, your number is optimistic and not trustworthy.

## 3. TF-IDF + logistic regression (the baseline)
- **TF** (term frequency): how often a word appears in this post.
- **IDF** (inverse document frequency): down-weights words that appear in almost every post ("the", "hai").
- So each post becomes a long vector: one number per word or letter pattern.
- **Character n-grams** (2-5 letters) also catch spelling variants: "besharam" and "besaram" share "bes", "ara", "ram".
- **Logistic regression** learns a weight for each feature; the weighted sum goes through a sigmoid to give P(hate).
- **Why have a baseline?** Without one, you can't say whether the fancy model is actually better.
- **Weakness:** no understanding of context or word order beyond 2 words. "I hate Mondays" and "I hate [group]" look similar.

## 4. Transformers and fine-tuning
- A **transformer** reads the whole sentence at once and uses **attention**: each word looks at
  every other word to decide what it means *in this context*.
- **Pre-training**: the model learns language by predicting masked-out words in huge amounts of text
  (that's what the "BERT" family does). No labels needed.
- **Fine-tuning**: add a small classification layer on top and train the whole model a little more
  on our labelled posts (3 epochs, learning rate 2e-5).
- **Tokenization**: text is split into sub-word pieces (e.g. "besharam" becomes "bes", "##har", "##am"). A model
  whose vocabulary was never trained on romanised Hindi splits Hinglish words into meaningless fragments.

## 5. Why MuRIL vs XLM-R
- **XLM-R**: Meta, 100 languages, CommonCrawl web text. Mostly native scripts.
- **MuRIL**: Google, 17 Indian languages, and it was **also trained on transliterated text**
  (Hindi written in Latin letters). That is exactly what Hinglish is.
- **Hypothesis:** MuRIL should help most on Hinglish. Your results tell you whether that's true.
- **Real example from this project** (tokenizing "yaar ye banda total besharam hai"):
  - MuRIL: `yaar | ye | banda | total | besh ##aram | hai` keeps "yaar" whole.
  - XLM-R: `ya ar | ye | banda | total | bes haram | hai` breaks "yaar" apart, and splits "besharam"
    (shameless) into "bes" + "haram", a *different* loaded word. Messy splits like this are one
    reason models struggle with Hinglish.

## 6. Zero-shot LLM (Gemini)
- **Zero-shot** = no training examples. We give a definition of hate speech in the prompt and ask for a label.
- **Pro:** no labelled data needed; understands sarcasm and context better.
- **Con:** its idea of "hate" may not match the dataset's annotation guidelines. So "low F1" can mean
  the LLM is wrong, or that **the labels follow a different definition**. That's why we read disagreements by hand.
- We turned off Gemini's safety filters because it has to read hateful text in order to label it.

## 7. Metrics: why macro-F1, not accuracy
- **Precision** (for "hate"): of the posts I called hate, how many really were?
- **Recall**: of the real hate posts, how many did I catch?
- **F1** = harmonic mean of precision and recall (it's high only if both are high).
- **Macro-F1** = average of F1 for "hate" and F1 for "not hate". With imbalanced data (Hinglish here is
  60% not-hate), a model that always says "not hate" gets 60% accuracy but a terrible macro-F1.
- HASOC (Prof. Mandl's shared task) uses macro-F1 as its official metric.

## 8. Error analysis (the "research" part)
Scores tell you *how much* a model fails; error analysis tells you *why*. You read 60 mistakes and tag
each with a category (wrong gold label, sarcasm, reporting on hate, spelling, ...). Then you count.
A sentence like "40% of Hinglish errors were actually label noise" is a **finding**.

## 9. Your honest limitations (say these before they ask)
- The dataset is merged from several sources with inconsistent label definitions.
- Labels are binary only. HASOC separates hate, offensive, and profane, which is more realistic.
- Gemini was tested on 300 posts, not all 8,852.
- One random seed; differences of 1 F1 point may be noise.

## 10. Likely interview questions
- *Why this dataset and not HASOC?* HASOC needs a data agreement; PRISM was open and has per-post
  language tags. Next step: re-run on HASOC. The code only needs a different loader.
- *What would you do with more time?* Separate hate from offensive; measure the amount of code-mixing
  per post and see whether errors increase with it; transliterate Hinglish to Devanagari before classifying.
- *What surprised you?* (Fill in from your results: probably the label noise.)
- *Why is Hindi (Devanagari) also hard, not just Hinglish?* Less training data per language, and
  TF-IDF features don't share anything between Devanagari and Latin script.
