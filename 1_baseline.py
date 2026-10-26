"""Model 1: TF-IDF + logistic regression (the "simple student").

TF-IDF turns each post into a vector of numbers: how often each word / letter
pattern appears, down-weighted if it appears in almost every post.
Logistic regression then learns one weight per pattern: positive weights push
towards "hate", negative weights towards "not hate".

We use two kinds of features:
  - word n-grams (1-2 words), e.g. "rape", "go back"
  - character n-grams (2-5 letters), e.g. "chut", "besh"; these help with
    Hinglish, where the same word is spelled many ways ("besharam", "besaram").
"""
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import load_split, macro_f1, print_scores, save_predictions, score_by_language

train, val, test = load_split("train"), load_split("val"), load_split("test")

word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)


def features(texts, fit=False):
    if fit:
        return hstack([word_vec.fit_transform(texts), char_vec.fit_transform(texts)]).tocsr()
    return hstack([word_vec.transform(texts), char_vec.transform(texts)]).tocsr()


X_train = features(train["text"], fit=True)
X_val, X_test = features(val["text"]), features(test["text"])

# Pick the regularisation strength C on the validation set (never on test).
best_c, best_f1 = None, -1
for c in [0.5, 1, 2, 4, 8]:
    clf = LogisticRegression(C=c, max_iter=2000)
    clf.fit(X_train, train["label"])
    f1 = macro_f1(val["label"], clf.predict(X_val))
    print(f"C={c:<4} val macro-F1={f1:.3f}")
    if f1 > best_f1:
        best_c, best_f1 = c, f1

print(f"best C={best_c}")
clf = LogisticRegression(C=best_c, max_iter=2000).fit(X_train, train["label"])

test["id"] = test.index
test["pred"] = clf.predict(X_test)
print_scores("tfidf_lr", score_by_language(test))
save_predictions(test, "tfidf_lr")
