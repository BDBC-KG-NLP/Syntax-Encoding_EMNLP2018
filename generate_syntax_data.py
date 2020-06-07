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

os.environ['STANFORD_PARSER'] = '/home/LAB/huzy/stanford/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/LAB/huzy/stanford/stanford-parser-3.9.1-models.jar'

parser = StanfordParser(model_path='/home/LAB/huzy/stanford/englishPCFG.ser.gz')
token = StanfordTokenizer("/home/LAB/huzy/stanford/stanford-parser.jar")


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


def write_label_to_file(i, num1, ll):
    with open("data/new/author" + str(i) + "/" + str(num1) + ".txt", "a+") as f1:
        for word_labels in ll:
            word_label = word_labels.split("@@")
            for label in word_label[:-1]:
                if label == "." or label == "''":
                    continue
                f1.write(label + " ")
            f1.write("\n")


path = []
ll = []


def parse1(author_label):
    global ll
    global path
    num = []
    paper_sentence_num = []
    for i in range(author_label, author_label + 1):
        # get the author's documents per author
        with open("data/word_data/all/author" + str(i), "r") as f:
            documents = f.readlines()
            num1 = 0
            # per paper

            for document in documents:
                num1 += 1
                sentences = []

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
                            write_label_to_file(i, num1, ll)
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
                            write_label_to_file(i, num1, ll)

                        except:
                            print("-----------------------------------------")
                            exception_print(i, num1, words)
                            print('traceback.format_exc():\n%s' % traceback.format_exc())
                            print("-----------------------------------------\n")


if __name__ == '__main__':
    # parse1(0)

    procs = []
    print("begin")
    queue = Queue()
    author_list = [40]

    for i in author_list:
        # for i in range(52,63):
        p = Process(target=parse1, args=(i,))
        p.start()
        procs.append(p)

    while len(procs) > 0:
        p = procs[0]
        p.join()
        procs.remove(p)
