import os
import nltk.data
import nltk
from nltk.parse.stanford import StanfordParser
from nltk.tokenize.stanford import StanfordTokenizer
from multiprocessing import Process, Queue
import pandas as pd

os.environ['STANFORD_PARSER'] = '/home/LAB/yuanhl/stanford/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = '/home/LAB/yuanhl//stanford/stanford-parser-3.9.2-models.jar'

parser = StanfordParser()
token = StanfordTokenizer("/home/LAB/yuanhl/stanford/stanford-parser.jar")


def split1(document):
    tokensen = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokensen.tokenize(document)
    return sentences


def split2(document):
    sentences = document.split(".")
    return sentences


def back_track(tree):
    path.append(tree.label())
    for child in tree:
        if type(child) == nltk.tree.Tree and len(child) != 0:
            back_track(child)
        elif type(child) == str or (type(child) == nltk.tree.Tree and len(child) != 0):
            path.append(child)
            ll.append("@@".join(path))
            path.pop()
            path.pop()


def write_label_to_file(index, ll):
    with open(parse_path.format(dataset, author_num, train_flag, str(index)), "a+") as f1:
        for word_labels in ll:
            word_label = word_labels.split("@@")
            for label in word_label[:-1]:
                if label == "." or label == "''":
                    continue
                f1.write(label + " ")
            f1.write("\n")


path = []
ll = []


def parse1(index_list, train_data):
    global ll
    global path
    for index, document in zip(index_list, train_data.iloc[index_list][2]):
        try:
            sentences = split1(document)
        except Exception as e:
            print(e)
            sentences = split2(document)

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
                write_label_to_file(index, ll)
            except Exception as e:
                print(e)


dataset = 'ag_news'
author_num = '4'
train_flag = 'test'
file_path = './data/{}/{}/word_data/{}.csv'
parse_path = './data/{}/{}/parse_data/{}/{}.txt'

if __name__ == '__main__':
    procs = []
    print("begin")
    queue = Queue()
    train_data = pd.read_csv('./data/{}/{}/word_data/{}.csv'.format(dataset, author_num, train_flag), header=None)
    num_news = len(train_data.values)
    print(num_news)
    indexs_list = range(num_news)
    progress_num = 10
    per_progress = int(num_news/progress_num)
    for i in range(progress_num):
        if i + 1 == progress_num:
            index_list = list(indexs_list[i * per_progress: num_news])
        else:
            index_list = list(indexs_list[i * per_progress: (i + 1) * per_progress])

        p = Process(target=parse1, args=(index_list, train_data))
        p.start()
        procs.append(p)

    while len(procs) > 0:
        p = procs[0]
        p.join()
        procs.remove(p)
