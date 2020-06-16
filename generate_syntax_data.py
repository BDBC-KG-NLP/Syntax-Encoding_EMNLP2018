# -*-coding:utf-8-*-
import os
import sys
import traceback
import nltk.data  # this is for the function -- split sentence
import nltk
from nltk.parse.stanford import StanfordParser
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tag import StanfordPOSTagger
from collections import Counter
import threading
from multiprocessing import Process, Queue
import chardet
import datetime

os.environ['STANFORD_PARSER'] = '/home/huzy/stanford/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/huzy/stanford/stanford-parser-3.9.2-models.jar'

parser = StanfordParser()
token = StanfordTokenizer("/home/huzy/stanford/stanford-parser.jar")


# split sentences in document
# the result is sentences list
def split1(document):
    tokensen = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokensen.tokenize(document)
    return sentences


def split2(document):
    sentences = document.split(".")
    return sentences


def back_track(tree):
    path.append(tree.label())
    # print(path)
    for child in tree:
        if type(child) == nltk.tree.Tree and len(child) != 0:
            back_track(child)
        elif type(child) == str or (type(child) == nltk.tree.Tree and len(child) != 0):
            path.append(child)
            # the ll is a list which includes the labels of a word
            ll.append("@@".join(path))
            # print(ll)
            path.pop()
            path.pop()


def exception_print(i, num1, words):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("\n")
    print("error:author " + str(i) + " document " + str(num1) + "\n")
    print(words)
    print("\n")


def write_label_to_file(dataset, i, num1, ll, num_authors=None):
    file = dataset
    if num_authors is not None:
        file += ('/' + str(num_authors))
    if not os.path.exists(file + "/parse_data/author" + str(i)):
        os.mkdir(file + "/parse_data/author" + str(i))
    with open(file + "/parse_data/author" + str(i) + "/" + str(num1) + ".txt", "a+") as f1:
        for word_labels in ll:
            word_label = word_labels.split("@@")
            for label in word_label[:-1]:
                if label == "." or label == "''":
                    continue
                f1.write(label + " ")
            f1.write("\n")


path = []
ll = []


def parse1(dataset, author_list, num_authors=None):
    global ll
    global path

    file = dataset
    if num_authors is not None:
        file += ('/' + str(num_authors))
    # get the author's documents per author
    for i in author_list:
        with open(file + "/word_data/all/author" + str(i), "r") as f:
            documents = f.readlines()
            num1 = 0

            for document in documents:
                num1 += 1
                try:  # can split or not,if the answer is yes,continue,if no,exception
                    sentences = split1(document)
                    for sentence in sentences:

                        words = token.tokenize(sentence)
                        if len(words) > 120:
                            continue
                        try:
                            results = parser.parse(words)
                            root = list(results)[0]
                            ll = []
                            path = []
                            back_track(root)
                            write_label_to_file(dataset, i, num1, ll, num_authors)
                        except:
                            print("-----------------------------------------")
                            exception_print(i, num1, words)
                            print('traceback.format_exc():\n%s' % traceback.format_exc())
                            print("-----------------------------------------\n")

                except:
                    sentences = split2(document)
                    print("--------------------------")
                    print("false")
                    print("error_split:author " + str(i) + " document " + str(num1) + "\n")
                    print(document)
                    print("--------------------------")

                    for sentence in sentences:
                        if sentence == " " or sentence == "\r\n" or sentence == '' or sentence == '\n':
                            continue
                        words = token.tokenize(sentence)

                        if len(words) > 120:
                            continue
                        try:
                            results = parser.parse(words)
                            root = list(results)[0]
                            ll = []
                            path = []
                            back_track(root)
                            write_label_to_file(dataset, i, num1, ll, num_authors)

                        except:
                            print("-----------------------------------------")
                            exception_print(i, num1, words)
                            print('traceback.format_exc():\n%s' % traceback.format_exc())
                            print("-----------------------------------------\n")


if __name__ == '__main__':
    # parse1(0)
    dataset = 'imdb'
    # num_authors = 62
    procs = []
    print("begin")
    queue = Queue()

    # parse1(dataset, [1])
    authorList = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52,
                  53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

    num_authors = len(authorList)
    progress_num = 5
    per_progress = int(num_authors / progress_num)
    for i in range(progress_num):
        if i + 1 == progress_num:
            author_list = list(authorList[i * per_progress: num_authors])
        else:
            author_list = list(authorList[i * per_progress: (i + 1) * per_progress])

        print(author_list)
        p = Process(target=parse1, args=(dataset, author_list))
        p.start()
        procs.append(p)

    while len(procs) > 0:
        p = procs[0]
        p.join()
        procs.remove(p)
