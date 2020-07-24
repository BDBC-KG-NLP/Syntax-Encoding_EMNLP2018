# -*- coding:utf-8 -*-
__author__ = 'shrprasha'

import itertools

import os
from collections import Counter
import nltk
import numpy
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import math
import utils
import codecs
from nltk.tag import pos_tag_sents
from nltk.corpus import twitter_samples

# from nltk.tokenize.stanford import StanfordTokenizer
# os.environ['STANFORD_PARSER'] = '/home/LAB/huzy/stanford/stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = '/home/LAB/huzy/stanford/stanford-parser-3.9.1-models.jar'

# token = StanfordTokenizer("/home/LAB/huzy/stanford/stanford-parser.jar")


seed = 7
numpy.random.seed(seed)


# created in order to store word_to_idx, for eg. when we have validation and test sets
# truncates any test data > max_len of train
class Features:

    def __init__(self, feature_type, min_df=0):
        # print feature_type
        if feature_type == "word":
            self.feature_fxn = get_char_ngrams_fxn(ngram=1)
            # self.feature_fxn = ark_tweet_tokenizer
        elif feature_type == "char_1":
            self.feature_fxn = get_char_ngrams_fxn(ngram=1)
        elif feature_type == "char_2":
            self.feature_fxn = get_char_ngrams_fxn(ngram=2)
        elif feature_type == "char_3":
            self.feature_fxn = get_char_ngrams_fxn(ngram=3)
        else:
            self.feature_fxn = get_char_ngrams_fxn(ngram=3)
        self.min_df = min_df

    def fit(self, X):
        X_tokenized = [self.feature_fxn(t) for t in X]
        all_words_ctr = Counter(itertools.chain.from_iterable(X_tokenized))
        if self.min_df:
            print("before filter by threshold:", len(all_words_ctr))
            utils.ctr_remove_below_threshold(all_words_ctr, self.min_df)
            print("after filter by threshold:", len(all_words_ctr))
        all_words = ["<PAD>"] + list(all_words_ctr.keys())  # reserving 0 for padding
        self.feature_map = dict(zip(all_words, range(len(all_words))))
        self.max_len = max(len(s) for s in X_tokenized)
        print("end....")

    def transform1(self, X):
        # word
        X_tokenized1 = [self.feature_fxn(t) for t in X]

        # POS label
        tweets_tagged = pos_tag_sents(X_tokenized1)
        X_tokenized = []
        for i in range(len(tweets_tagged)):
            X_tokenized.append([])
            # i 是第几篇文章
            for j in range(len(tweets_tagged[i])):
                X_tokenized[i].append(tweets_tagged[i][j][1])
        # print(X_tokenized[2])
        # print("------------------------")
        X = []
        for tweet in X_tokenized:
            X.append([self.feature_map[w] for w in tweet if w in self.feature_map])
        X = sequence.pad_sequences(X, maxlen=self.max_len, padding='post')
        return X

    def transform(self, X):
        print(X[0])
        X_tokenized1 = [self.feature_fxn(t) for t in X]
        print(X_tokenized1[0])
        X_tokenized = X_tokenized1
        # to divide train and test
        print("------------------------")
        print("\n")
        X = []
        for tweet in X_tokenized:
            X.append([self.feature_map[w] for w in tweet if w in self.feature_map])
        X = sequence.pad_sequences(X, maxlen=self.max_len, padding='post')
        return X


# def ark_tweet_tokenizer(text):
#     return twokenize.tokenizeRawTweetText(text)


def convert_data_for_nn_models(X_train, X_test, Y_train, Y_test, val_size=0.2):
    le = preprocessing.LabelEncoder()
    # !!!!!!!if val_size is 0. no val
    if val_size:
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, stratify=Y_train, test_size=val_size,
                                                          random_state=190)
        print(Y_train)

        le.fit(Y_train)
        Y_train = list(le.transform(Y_train))

        print("!!!!")
        Y_val = list(le.transform(Y_val))
        Y_test = list(le.transform(Y_test))
        return X_train, np_utils.to_categorical(Y_train), X_val, np_utils.to_categorical(
            Y_val), X_test, np_utils.to_categorical(Y_test)
    else:
        le.fit(Y_train)
        # you can find the coopration of author and the label
        Y_train = list(le.transform(Y_train))
        Y_test = list(le.transform(Y_test))

        return X_train, np_utils.to_categorical(Y_train), X_test, np_utils.to_categorical(Y_test)


def get_authors_train_test(rootdir, authors, val_size=0.2, train_ratio=0.5):
    X_train, X_test, Y_train, Y_test = get_all_authors_create_train_test(rootdir, authors, train_ratio)

    return convert_data_for_nn_models(X_train, X_test, Y_train, Y_test, val_size)


def get_all_authors_create_train_test(rootdir, authors, train_ratio):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for author in authors:
        train_reviews, test_reviews = get_author_text_train_test(rootdir, author, train_ratio=train_ratio)
        X_train.extend(train_reviews)
        X_test.extend(test_reviews)
        Y_train.extend([author] * len(train_reviews))
        Y_test.extend([author] * len(test_reviews))
    return X_train, X_test, Y_train, Y_test


def get_author_text_train_test(rootdir, author_fname, train_ratio=0.5):
    posts = list(utils.get_lines_in_file_small(os.path.join(rootdir, author_fname)))
    train_posts_count = int(math.ceil(len(posts) * train_ratio))
    return posts[:train_posts_count], posts[train_posts_count:]


def get_char_ngrams_fxn(ngram):
    def char_ngrams(text):
        return [text[i:i + ngram] for i in range(len(text) - ngram + 1)]

    return char_ngrams


'''
def get_word():
    def sp(text):
        ## text is per docu
        a = text.split(" ")
        fi =[]
        for i in range(len(a)):
            b = a[i].split(",")
            b1 = a[i].split(".")
            if len(b)>1:
                for j in range(len(b)):
                    fi.append(b[j])

            if len(b1)>1:
                for k in range(len(b1)):
                    fi.append(b1[k])
        else:
            fi.append(a[i])

        return fi
    return sp
'''

# def get_word1():
#     def sp(text):
#         tokensen = nltk.data.load('tokenizers/punkt/english.pickle')
#         senten1 = tokensen.tokenize(text)
#         return senten1
#     return sp


# def get_word():
#     def sp(text):
#         words = token.tokenize(text)
#         return words
#     return sp


'''
words = token.tokenize(sentence)

os.environ['STANFORD_PARSER'] = '/home/LAB/huzy/stanford/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/LAB/huzy/stanford/stanford-parser-3.9.1-models.jar'

token = StanfordTokenizer("/home/LAB/huzy/stanford/stanford-parser.jar")
'''
