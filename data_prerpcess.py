# -*- coding:utf-8 -*-
import os
import codecs
import numpy as np
from keras.utils import np_utils
import traceback


def get_blogs_train_and_test(authors_count, min_doc=None):
    path = 'blogs/'+str(authors_count)+'/{}/author{}'
    file = 'blogs/' + str(authors_count) + '/{}/author{}/{}_syntax.txt'
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    label_list = []
    num = 0
    for i in range(1, authors_count + 1):
        print(i)
        doc_num_train = len(list(filter(lambda x: '_' not in str(x), os.listdir(path.format('train', str(i))))))
        if min_doc is not None:
            # blogs数据集较大容易，限制取得文本数
            doc_num_train = min(min_doc, doc_num_train)

        num += doc_num_train
        for j in range(1, doc_num_train + 1):
            try:
                document = []
                with codecs.open(file.format('train', str(i), str(j)), "r", "latin1") as f:
                    words = f.readlines()
                    for index, word in enumerate(words):

                        # 限制每个文本取得字节数
                        if index >= 500:
                            break

                        w_lsit = []
                        if word != "\n":
                            words_label = word.split(" ")
                            for w in words_label:
                                if w == '``' or w == '#' or w == ':' or w == '' or w == " ":
                                    continue
                                else:
                                    if w not in label_list:
                                        label_list.append(w)
                                    w_lsit.append(w)
                            document.append(w_lsit)
                X_train.append(document)
                Y_train.append(i - 1)
            except Exception as e:
                print(e)
                document = []
                X_train.append(document)
                Y_train.append(i - 1)
                print("train:author" + str(i) + " " + str(j) + "document")

        doc_num_test = len(list(filter(lambda x: '_' not in str(x), os.listdir(path.format('test', str(i))))))
        for j in range(1, doc_num_test + 1):
            try:
                document = []
                with codecs.open(file.format('train', str(i), str(j)),"r", "latin1") as f:
                    words = f.readlines()
                    for index, word in enumerate(words):
                        if index >= 500:
                            break
                        w_lsit = []
                        if word != "\n":
                            words_label = word.split(" ")
                            for w in words_label:
                                if w == '``' or w == '#' or w == ':' or w == '' or w == ' ':
                                    continue
                                else:
                                    w_lsit.append(w)
                            document.append(w_lsit)

                X_test.append(document)
                Y_test.append(i - 1)
            except Exception as e:
                print(e)
                document = []
                X_test.append(document)
                Y_test.append(i - 1)
                print("test:author" + str(i) + " " + str(j) + "document")

    dic = {}
    i = 0
    for label in label_list:
        i += 1
        dic[label] = i

    document_length = []
    for i in range(len(X_train)):
        document_length.append(len(X_train[i]))

    label_dic = len(label_list)
    print(num)
    return X_train, Y_train, X_test, Y_test, dic, sorted(document_length)[-1], label_dic


def get_CCAT_train_and_test(dataset, authors_count):
    perDocNum = 100
    if dataset == 'CCAT':
        perDocNum = 100
        file = 'CCAT/' + str(authors_count) + '/parse_data/author{}/{}.txt'
    elif dataset == 'imdb':
        perDocNum = 1000
        file = 'imdb/parse_data/author{}/{}.txt'
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    label_list = []
    num = 0
    for i in range(1, authors_count + 1):
        print(i)
        num += perDocNum
        for j in range(1, int(perDocNum/2)+1):
            try:
                document = []
                with codecs.open(file.format(str(i), str(j)), "r", "latin1") as f:
                    words = f.readlines()
                    for index, word in enumerate(words):
                        w_lsit = []
                        if word != "\n":
                            words_label = word.split(" ")
                            for w in words_label:
                                if w == '``' or w == '#' or w == ':' or w == '' or w == " ":
                                    continue
                                else:
                                    if w not in label_list:
                                        label_list.append(w)
                                    w_lsit.append(w)
                            document.append(w_lsit)
                X_train.append(document)
                Y_train.append(i - 1)
            except Exception as e:
                print(e)
                document = []
                X_train.append(document)
                Y_train.append(i - 1)
                print("train:author" + str(i) + " " + str(j) + "document")

        for j in range(int(perDocNum/2)+1, perDocNum+1):
            try:
                document = []
                with codecs.open(file.format(str(i), str(j)), "r", "latin1") as f:
                    words = f.readlines()
                    for index, word in enumerate(words):
                        w_lsit = []
                        if word != "\n":
                            words_label = word.split(" ")
                            for w in words_label:
                                if w == '``' or w == '#' or w == ':' or w == '' or w == ' ':
                                    continue
                                else:
                                    w_lsit.append(w)
                            document.append(w_lsit)

                X_test.append(document)
                Y_test.append(i - 1)
            except Exception as e:
                print(e)
                document = []
                X_test.append(document)
                Y_test.append(i - 1)
                print("test:author" + str(i) + " " + str(j) + "document")

    dic = {}
    i = 0
    for label in label_list:
        i += 1
        dic[label] = i

    document_length = []
    for i in range(len(X_train)):
        document_length.append(len(X_train[i]))

    label_dic = len(label_list)

    # Y_train is the label of train data
    # Y_test is the label of train data
    print(num)
    return X_train, Y_train, X_test, Y_test, dic, sorted(document_length)[-1], label_dic


def transform(X, dic, length):
    print(length)
    X_label = np.zeros((len(X), length, 150), dtype=np.int32)
    print(X_label.shape)
    for i in range(len(X)):
        for j in range(len(X[i])):
            try:
                if len(X[i][j]) < 150:
                    for k in range(len(X[i][j])):
                        X_label[i][j][k] = dic[X[i][j][k]]
            except:
                print('traceback.format_exc():\n%s' % traceback.format_exc())
    return X_label


def transform_test(X):
    return np_utils.to_categorical(np.array(X))


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test, dic, length, label_dic = get_blogs_train_and_test(3)
    X_train_single = transform(X_train, dic, length)
    X_test_single = transform(X_test, dic, length)

    Y_train, Y_test = transform_test(Y_train), transform_test(Y_test)
