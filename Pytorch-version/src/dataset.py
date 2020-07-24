import math
# from keras.utils import np_utils
from config import *
from src.utils2 import *
from src.get_data import *
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class SyntaxDataset(Dataset):
    """
    加载数据集，文本数据，语法编码数据
    返回 文本向量，语法编码向量，作者编号
    """

    def __init__(self, dataset, author_num, train_ratio=0.5, max_doc=100, max_len=None):
        """
        加载训练集与测试集
        :param dataset: 需要加载的数据集名称
        :param author_num: 数据集的作者规模，根据数据集情况
        :param train_ratio: imdb与CCAT训练集与测试集放在一起，按照比例划分训练集与测试集
        :param max_doc: blogs数据集较大，为防止超内存，可限制加载每位作者的文档数
        :param max_len: 限制加载语法编码文件的行数
        """
        super(SyntaxDataset, self).__init__()
        self.train_flag = True  # 切换训练集与测试集

        if dataset == 'ag_news' or dataset == 'dbpedia':
            text_train, syntax_train, Y_train, text_test, syntax_test, Y_test, label_list = \
                get_news_data(dataset, author_num, max_len=max_len)
        elif dataset == 'blogs':
            text_train, syntax_train, Y_train, text_test, syntax_test, Y_test, label_list = \
                get_blogs_data(dataset, author_num, max_doc=max_doc, max_len=max_len)
        else:
            text_train, syntax_train, Y_train, text_test, syntax_test, Y_test, label_list = \
                get_others_data(dataset, author_num, train_ratio=train_ratio, max_len=max_len)

        print('load data end ...')
        # 语法embeding的时候需要
        self.syntax_len = len(label_list)
        # 语法编码字符对应数字 map
        label_dic = {}
        i = 0
        for label in label_list:
            i += 1
            label_dic[label] = i

        # 获取最长文本的字符数
        max_len = 0
        tmp = []
        tmp.extend(syntax_train)
        tmp.extend(syntax_test)
        for item in tmp:
            max_len = max(max_len, len(item))

        # 将文档语法编码字符转成数字
        self.syntax_train_code = np.array(transform(syntax_train, label_dic, max_len), np.int64)
        self.syntax_test_code = np.array(transform(syntax_test, label_dic, max_len), np.int64)
        print('syntax code map end ...')

        tmp = []
        tmp.extend(text_train)
        tmp.extend(text_test)
        single_feats = Features()
        single_feats.fit(tmp)

        # 文本embeding的时候需要
        self.char_len = len(single_feats.feature_map)
        # 将文档字符转成数字
        self.text_train_code = np.array(single_feats.transform(text_train), np.int64)
        self.text_test_code = np.array(single_feats.transform(text_test), np.int64)
        print('text code map end ...')

        self.train_label = Y_train
        self.test_label = Y_test

    def __len__(self):
        if self.train_flag:
            return len(self.train_label)
        else:
            return len(self.test_label)

    def __getitem__(self, index):
        if self.train_flag:
            return self.text_train_code[index], self.syntax_train_code[index], self.train_label[index]
        else:
            return self.text_test_code[index], self.syntax_test_code[index], self.test_label[index]


def get_syntax_data(path, label_list, max_len=None):
    syntax_data = []
    try:
        with open(path, "r", encoding="latin1") as f:
            words = f.readlines()
            for index, word in enumerate(words):
                if max_len is not None and index >= max_len:
                    continue
                w_lsit = []
                if word != "\n":
                    words_label = str(word).strip().split(" ")
                    for w in words_label:
                        if w == '``' or w == '#' or w == ':' or w == '' or w == " ":
                            continue
                        else:
                            if w not in label_list:
                                label_list.append(w)
                            w_lsit.append(w)
                    syntax_data.append(w_lsit)
    except Exception as e:
        print(e)
    return syntax_data


def get_news_data(dataset, author_num, max_len=None):
    text_train = []
    syntax_train = []
    Y_train = []
    text_test = []
    syntax_test = []
    Y_test = []
    label_list = []

    for train_flag in ['test', 'train']:
        print(train_flag)
        all_data = pd.read_csv(news_text_file.format(dataset, author_num, train_flag), header=None)
        for index, record in enumerate(all_data.values):
            if index % 1000 == 0:
                print(index)
            if train_flag == 'train':
                text_train.append(record[2])
                Y_train.append(int(record[0])-1)
                syntax_train.append(
                    get_syntax_data(
                        news_syntax_file.format(dataset, author_num, 'train', index), label_list, max_len=max_len))
            else:
                text_test.append(record[2])
                Y_test.append(int(record[0])-1)
                syntax_test.append(
                    get_syntax_data(
                        news_syntax_file.format(dataset, author_num, 'test', index), label_list, max_len=max_len))

    # Y_train = np_utils.to_categorical(Y_train)
    # Y_test = np_utils.to_categorical(Y_test)

    return text_train, syntax_train, Y_train, text_test, syntax_test, Y_test, label_list


def get_blogs_data(dataset, author_num, max_doc=None, max_len=None):
    text_train = []
    syntax_train = []
    Y_train = []
    text_test = []
    syntax_test = []
    Y_test = []
    label_list = []

    for i in range(1, author_num + 1):
        print("get the document of author {}".format(i))
        # 获取训练数据
        doc_num_train = len(
            list(filter(lambda x: '_' not in str(x), os.listdir(blog_data.format(dataset, author_num, 'train', i)))))
        if max_doc is not None:
            doc_num_train = min(max_doc, doc_num_train)
        for j in range(1, int(doc_num_train + 1)):
            with open(blog_text_file.format(dataset, author_num, 'train', i, j), "r", encoding='latin1') as f:
                document = f.read()
                document = document[:-1]
                text_train.append(document)
                Y_train.append(i - 1)

            syntax_train.append(
                get_syntax_data(blog_syntax_file.format(dataset, author_num, 'train', i, j), label_list,
                                max_len=max_len))

        # 获取测试数据
        doc_num_test = len(
            list(filter(lambda x: '_' not in str(x), os.listdir(blog_data.format(dataset, author_num, 'test', i)))))

        for j in range(1, int(doc_num_test + 1)):
            with open(blog_text_file.format(dataset, author_num, 'test', i, j), "r", encoding='latin1') as f:
                document = f.read()
                document = document[:-1]
                text_test.append(document)
                Y_test.append(i - 1)
            syntax_test.append(
                get_syntax_data(blog_syntax_file.format(dataset, author_num, 'test', i, j), label_list, max_len=max_len))

    # Y_train = np_utils.to_categorical(Y_train)
    # Y_test = np_utils.to_categorical(Y_test)

    return text_train, syntax_train, Y_train, text_test, syntax_test, Y_test, label_list


def get_others_data(dataset, author_num, train_ratio=0.5, max_len=None):
    text_train = []
    syntax_train = []
    Y_train = []
    text_test = []
    syntax_test = []
    Y_test = []
    label_list = []
    for i in range(1, author_num + 1):
        print("get the document of author {}".format(i))
        text_data = []
        syntax_data = []
        Y_data = []
        with open(text_file.format(dataset, author_num, i), "r", encoding='latin1') as f:
            for line in f.readlines():
                document = line[:-1]
                text_data.append(document)
                Y_data.append(i - 1)

        doc_num = len(text_data)
        for j in range(1, doc_num + 1):
            syntax_data.append(
                get_syntax_data(syntax_file.format(dataset, author_num, i, j), label_list, max_len=max_len))

        Y_label = Y_data  # np_utils.to_categorical(Y_data, num_classes=author_num)
        pos_index = int(math.ceil(doc_num * train_ratio))
        # 获取训练数据
        text_train.extend(text_data[0:pos_index])
        syntax_train.extend(syntax_data[0:pos_index])
        Y_train.extend(Y_label[0:pos_index])
        # 获取测试数据
        text_test.extend(text_data[pos_index:])
        syntax_test.extend(syntax_data[pos_index:])
        Y_test.extend(Y_label[pos_index:])

    return text_train, syntax_train, Y_train, text_test, syntax_test, Y_test, label_list


def transform(syntax_data, dic, length):
    """
    将文档的语法编码字符转换位对应的数字
    :param syntax_data: 全部文档的语法编码字符
    :param dic: 字符编码与数字map
    :param length: 文档包含的最多字符数
    :return:
    """
    syntax_code = np.zeros((len(syntax_data), length, 150), dtype=np.int32)
    print(syntax_code.shape)
    for i in range(len(syntax_data)):
        for j in range(len(syntax_data[i])):
            if len(syntax_data[i][j]) < 150:
                for k in range(len(syntax_data[i][j])):
                    syntax_code[i][j][k] = dic[syntax_data[i][j][k]]

    return syntax_code


if __name__ == '__main__':
    test_dataset = SyntaxDataset('CCAT', 10)
    test_params = {"batch_size": 20, "shuffle": False, "drop_last": True}
    test_generator = DataLoader(test_dataset, **test_params)
    test_dataset.train_flag = False
    for text_test, syntax_test, label_test in test_generator:
        print(text_test.size())
        print(syntax_test.size())
        print(label_test)
        break
