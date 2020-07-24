import itertools
from collections import Counter
from keras.preprocessing import sequence
from src.utils2 import *


class Features:

    def __init__(self, feature_type=None, min_df=0):
        if feature_type == "char_1":
            self.feature_fxn = get_char_ngrams_fxn(ngram=1)
        elif feature_type == "char_2":
            self.feature_fxn = get_char_ngrams_fxn(ngram=2)
        elif feature_type == "char_3":
            self.feature_fxn = get_char_ngrams_fxn(ngram=3)
        else:
            self.feature_fxn = get_char_ngrams_fxn(ngram=1)
        self.min_df = min_df

    def fit(self, X):
        X_tokenized = [self.feature_fxn(t) for t in X]
        all_words_ctr = Counter(itertools.chain.from_iterable(X_tokenized))
        if self.min_df:
            print("before filter by threshold:", len(all_words_ctr))
            ctr_remove_below_threshold(all_words_ctr, self.min_df)
            print("after filter by threshold:", len(all_words_ctr))
        all_words = ["<PAD>"] + list(all_words_ctr.keys())
        self.feature_map = dict(zip(all_words, range(len(all_words))))
        self.max_len = max(len(s) for s in X_tokenized)

    def transform(self, X):
        X_tokenized1 = [self.feature_fxn(t) for t in X]
        X_tokenized = X_tokenized1
        X = []
        for tweet in X_tokenized:
            X.append([self.feature_map[w] for w in tweet if w in self.feature_map])
        X = sequence.pad_sequences(X, maxlen=self.max_len, padding='post')
        return X


def get_char_ngrams_fxn(ngram):
    def char_ngrams(text):
        return [text[i:i + ngram] for i in range(len(text) - ngram + 1)]

    return char_ngrams
