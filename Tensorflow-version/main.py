# -*- coding:utf-8 -*-
from __future__ import division

__author__ = 'zhiyuan_hu'

import time
import pandas as pd
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Embedding, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D
from keras.layers.core import Lambda, Reshape
from keras.layers.noise import AlphaDropout
from keras.models import Model
from keras.optimizers import Adam
import os
from get_data import Features, get_authors_train_test
import utils
from keras.engine.topology import Layer

import argparse
import time
from keras.utils import np_utils
from data_prerpcess import *
from Syntax_Embeddings import Syntax_Embedding
import sys

seed = 7
np.random.seed(seed)
np.set_printoptions(threshold=np.inf)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)


datasets = ['emails', 'imdb', 'blogs', 'twitter', 'reddit', 'CCAT']
classif_id = 'CNN+gram'
parser = argparse.ArgumentParser(description=classif_id)
parser.add_argument('--num_authors', required=True, type=int, default=10, help='the number of authors')
parser.add_argument('--datasets', required=True, choices=datasets, help='the datasets')
parser.add_argument('--type', nargs='?', choices=['char_3', 'word'])
# parser.add_argument('--results', required=True, nargs='?',  help='the datasets')


args = parser.parse_args()

authors_count = args.num_authors
results_opdir = "best"
dataset = args.datasets

print(dataset)

authors_count = int(authors_count)
train_ratio = 0.5
if authors_count == 10 or authors_count == 50 or authors_count == 8:
    train_ratio = 0.5
if authors_count == 62:
    train_ratio = 0.9


def max_1d(X):
    return K.max(X, axis=1)


class AttLayer(Layer):
    # def __init__(self, **kwargs):
    #     self.init = initializers.get(init)
    def __init__(self, **kwargs):
        # self.output_dim = output_dim
        super(AttLayer, self).__init__(**kwargs)

    # input is (?,50) or (?,2500)
    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1', shape=(150, 1), initializer='uniform', trainable=True)
        self.W2 = self.add_weight(name='W2', shape=(7500, 1), initializer='uniform', trainable=True)
        # self.weight1 = self.add_weight(name='weight1',shape=(3),initializer='uniform',trainable=True)
        # self.weight2 = self.add_weight(name='weight2',shape=(3),initializer='uniform',trainable=True)

    def call(self, x):
        weight1 = weight2 = 0
        max1 = K.concatenate([x[0], x[1], x[2]], axis=1)
        max2 = K.concatenate([x[3], x[4], x[5]], axis=1)
        # max3 = K.concatenate([x[6],x[7],x[8]],axis=1)
        max0 = K.concatenate([max1, max2], axis=1)
        max01 = tf.reshape(max0, [-1, 150])
        exp1 = K.reshape(K.exp(K.dot(max01, self.W1)), [-1, 2])
        exp2 = tf.reshape(1 / K.sum(exp1, axis=1), [-1, 1])
        print(exp2)
        print("!!!!!!!!!!\n\n\n")
        weight1 = tf.multiply(exp1, exp2)
        print(weight1)
        print("!!!!!!!!!!\n\n\n")
        b = tf.gather(weight1, indices=[0], axis=1)
        b1 = tf.tile(b, [1, 150])
        b = tf.gather(weight1, indices=[1], axis=1)
        b2 = tf.tile(b, [1, 150])
        # b = tf.gather(weight1, indices=[2], axis=1)
        # b3 = tf.tile(b,[1,150])
        b11 = K.concatenate([b1, b2], axis=1)

        d = tf.multiply(max0, b11)

        max11 = K.concatenate([x[6], x[7], x[8]], axis=1)
        max21 = K.concatenate([x[9], x[10], x[11]], axis=1)
        # max31 = K.concatenate([x[15],x[16],x[17]],axis=1)
        max01 = K.concatenate([max11, max21], axis=1)
        max011 = tf.reshape(max01, [-1, 7500])
        exp11 = K.reshape(K.exp(K.dot(max011, self.W2)), [-1, 2])
        exp21 = tf.reshape(1 / K.sum(exp11, axis=1), [-1, 1])
        print(exp21)
        print("!!!!!!!!!!\n\n\n")
        weight2 = tf.multiply(exp11, exp21)
        print(weight2)
        print("!!!!!!!!!!\n\n\n")
        b = tf.gather(weight2, indices=[0], axis=1)
        b1 = tf.tile(b, [1, 7500])
        b = tf.gather(weight2, indices=[1], axis=1)
        b2 = tf.tile(b, [1, 7500])
        # b = tf.gather(weight2, indices=[2], axis=1)
        # b3 = tf.tile(b,[1,7500])
        b111 = K.concatenate([b1, b2], axis=1)

        d1 = tf.multiply(max01, b111)

        print(d1)
        print("d1!!!!!\n\n\n")

        # concat
        finall = K.concatenate([d, d1], axis=1)
        print(weight1)
        print("weight1\n\n\n")
        return finall

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 15300)


def gram_tri(gram):
    gram1 = K.transpose(gram)
    t = 0
    for i in range(50):
        for j in range(50):
            if j >= i:
                re_shape = K.reshape(gram1[i * 50 + j], (1, -1))
                if i == 0 and j == 0:
                    a1 = re_shape
                else:
                    a1 = K.concatenate([a1, re_shape], axis=0)
    a1 = K.transpose(a1)
    return a1


def gram(cnn):
    # gram3 = []
    gram = K.batch_dot(cnn, cnn, axes=[3, 3])
    gram = K.reshape(gram, (-1, 2500))
    return gram


def pass_by(gram_matrix):
    return gram_matrix


def concat(cnn1, cnn2, cnn3):
    # gram = tf.concat([cnn1,cnn2,cnn3],1)
    gram = K.concatenate((cnn1, cnn2, cnn3), axis=1)
    print(gram)
    return gram


def get_conv_layer(node_input, filter_tuple, activation_value):
    n_layers = len(filter_tuple)
    cnn_nodes = []
    gram_nodes = []
    gram_nodes1 = []
    for i in range(n_layers):
        cnn = Conv1D(nb_filter=filter_tuple[i][0], filter_length=filter_tuple[i][1], border_mode='causal',
                     activation=activation_value, subsample_length=1)(node_input)
        # max1 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(cnn)
        #
        # cnn2 = Conv1D(nb_filter=filter_tuple[i][0], filter_length=filter_tuple[i][1], border_mode='causal',
        #               activation=activation_value, subsample_length=1)(max1)

        # max2 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(cnn2)

        # cnn3 = Conv1D(nb_filter=filter_tuple[i][0], filter_length=filter_tuple[i][1],
        # border_mode='causal', activation=activation_value, subsample_length=1)(max2)

        cnn01 = Lambda(max_1d, output_shape=(filter_tuple[i][0],))(cnn)
        # cnn02 = Lambda(max_1d, output_shape=(filter_tuple[i][0],))(cnn2)

        # cnn03 = Lambda(max_1d, output_shape=(filter_tuple[i][0],))(cnn3)

        # gram_mat1 = Lambda(gram,output_shape=(filter_tuple[i][0]*filter_tuple[i][0],))(cnn)
        # gram_mat2 = Lambda(gram,output_shape=(filter_tuple[i][0]*filter_tuple[i][0],))(cnn2)
        # gram_mat3 = Lambda(gram,output_shape=(filter_tuple[i][0]*filter_tuple[i][0],))(cnn3)#

        gram_nodes.append(cnn01)
        # gram_nodes.append(cnn02)
        # gram_nodes.append(cnn03)

        # gram_nodes1.append(gram_mat1)
        # gram_nodes1.append(gram_mat2)
        # gram_nodes1.append(gram_mat3)

    # gram_nodes2 = []
    # for i in range(3):
    #     gram_nodes.append(gram_nodes1[i])
    print(gram_nodes)
    return gram_nodes
    # return cnn_nodes


def get_conv_layer_syntax(node_input, filter_tuple, activation_value):
    n_layers = len(filter_tuple)
    cnn_nodes = []
    gram_nodes = []
    gram_nodes1 = []
    for i in range(n_layers):
        cnn = Conv1D(nb_filter=filter_tuple[i][0], filter_length=filter_tuple[i][1], border_mode='causal',
                     activation=activation_value, subsample_length=1)(node_input)
        max1 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(cnn)

        cnn2 = Conv1D(nb_filter=filter_tuple[i][0], filter_length=filter_tuple[i][1], border_mode='causal',
                      activation=activation_value, subsample_length=1)(max1)

        # max2 = MaxPooling1D(pool_size=3, strides=None, padding='valid')(cnn2)

        # cnn3 = Conv1D(nb_filter=filter_tuple[i][0], filter_length=filter_tuple[i][1],
        # border_mode='causal', activation=activation_value, subsample_length=1)(max2)

        cnn01 = Lambda(max_1d, output_shape=(filter_tuple[i][0],))(cnn)
        # cnn02 = Lambda(max_1d, output_shape=(filter_tuple[i][0],))(cnn2)

        # cnn03 = Lambda(max_1d, output_shape=(filter_tuple[i][0],))(cnn3)

        # gram_mat1 = Lambda(gram,output_shape=(filter_tuple[i][0]*filter_tuple[i][0],))(cnn)
        # gram_mat2 = Lambda(gram,output_shape=(filter_tuple[i][0]*filter_tuple[i][0],))(cnn2)
        # gram_mat3 = Lambda(gram,output_shape=(filter_tuple[i][0]*filter_tuple[i][0],))(cnn3)#

        gram_nodes.append(cnn01)
        # gram_nodes.append(cnn02)
        # gram_nodes.append(cnn03)

        # gram_nodes1.append(gram_mat1)
        # gram_nodes1.append(gram_mat2)
        # gram_nodes1.append(gram_mat3)

    # gram_nodes2 = []
    # for i in range(3):
    #   gram_nodes.append(gram_nodes1[i])
    print(gram_nodes)
    return gram_nodes
    # return cnn_nodes


def get_cnn_model(no_of_labels, X_train, embedding_dims, char_to_idx, filter_tuple, dropout=0.25,
                  activation_value='relu', weights=None):
    # activation_value must be relu or tanh
    main_input = Input(shape=(X_train.shape[1],), dtype='int32', name='main_input')
    print(len(char_to_idx))
    print("!!!!!\n\n")
    x = Embedding(len(char_to_idx), embedding_dims, weights=weights, input_length=X_train.shape[1])(main_input)
    if dropout > 0:
        x = Dropout(dropout)(x)
    list_cnn_nodes = get_conv_layer(x, filter_tuple, activation_value)
    if len(list_cnn_nodes) > 1:
        list_cnn_nodes = layers.Concatenate()(list_cnn_nodes)
    else:
        list_cnn_nodes = list_cnn_nodes[0]
    # for i in range(len(list_cnn_nodes)):
    #     print(list_cnn_nodes[i])
    print("list_cnn_nodes\n")
    # Att = AttLayer()(list_cnn_nodes)
    if dropout > 0:
        list_cnn_nodes = AlphaDropout(dropout)(list_cnn_nodes)
    main_loss = Dense(no_of_labels, activation='softmax', name='main_output')(list_cnn_nodes)
    print("here\n\n\n")
    model = Model(input=main_input, output=main_loss)
    print("OK\n\n\n")
    return model


def plot_learning_curve(history, opfpath):
    import matplotlib
    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    print(history.history.keys())
    # print history.history['acc']
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(opfpath)


def train_and_test(X_train_word, X_train_syntax, Y_train, X_test_word, X_test_syntax, Y_test, nb_epochs, batch_size,
                   learning_rate, no_of_labels, embedding_dims_word, embedding_dims_syntax, char_to_idx, filter_tuple,
                   filter_tuple1, dropout, activation_value, length, label_dic, model_type, logdir, weights=None,
                   weights_pos=None):
    # activation_value must be relu or tanh
    bm_scores = True
    binary = False
    main_input = Input(shape=(X_train_word.shape[1],), dtype='int32', name='main_input')
    x = Embedding(len(char_to_idx), embedding_dims_word, weights=weights, input_length=X_train_word.shape[1])(
        main_input)
    if dropout > 0:
        x = Dropout(dropout)(x)
    list_cnn_nodes = get_conv_layer(x, filter_tuple, activation_value)
    if len(list_cnn_nodes) > 1:
        list_cnn_nodes = layers.Concatenate()(list_cnn_nodes)
    else:
        list_cnn_nodes = list_cnn_nodes[0]
    if dropout > 0:
        list_cnn_nodes = AlphaDropout(dropout)(list_cnn_nodes)

    # the start of  syntax ----------------    

    main_input_syntax = Input(shape=(X_train_syntax.shape[1], X_train_syntax.shape[2]), dtype='int32',
                              name='main_input_syntax')
    x_syntax = Syntax_Embedding(input_dim=label_dic, syntax_tree_label_dim=embedding_dims_syntax,
                                depth_of_syntax_tree=150, input_length=length)(main_input_syntax)

    if dropout > 0:
        x_syntax = Dropout(dropout)(x_syntax)
    list_cnn_nodes_syntax = get_conv_layer_syntax(x_syntax, filter_tuple1, activation_value)
    if len(list_cnn_nodes_syntax) > 1:
        list_cnn_nodes_syntax = layers.Concatenate()(list_cnn_nodes_syntax)
    else:
        list_cnn_nodes_syntax = list_cnn_nodes_syntax[0]
    if dropout > 0:
        list_cnn_nodes_syntax = AlphaDropout(dropout)(list_cnn_nodes_syntax)

    # the end of syntax    -----------------

    ###how to concat
    list_final = []
    list_final.append(list_cnn_nodes)
    list_final.append(list_cnn_nodes_syntax)
    list_cnn_nodes = layers.Concatenate()(list_final)

    main_loss = Dense(no_of_labels, activation='softmax', name='main_output')(list_cnn_nodes)

    model = Model(inputs=[main_input, main_input_syntax], output=main_loss)
    print("model")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # checkpointer = ModelCheckpoint(save_best_only=True, monitor='val_acc', verbose=1, filepath=os.path.join(logdir, model_type+timestamp+".hdf5"), )
    # dict_callbacks = {'checkpointer':checkpointer}
    # , callbacks=dict_callbacks.values()
    adam = Adam(lr=learning_rate)
    model.summary()
    if binary:
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    history = model.fit([X_train_word, X_train_syntax], Y_train, batch_size=batch_size, nb_epoch=300,
                        validation_data=([X_test_word, X_test_syntax], Y_test))

    open(os.path.join(logdir, model_type + timestamp + ".json"), 'w').write(model.to_json())

    model.summary()


def get_author_names_all_authors(author_lists_fpath):
    return [line.strip() for line in utils.get_lines_in_file(author_lists_fpath)]


def get_data_for_nns():
    if dataset == 'CCAT':
        data_rootdir = "CCAT/"+str(authors_count)+"/word_data/all"
    elif dataset == 'imdb':
        data_rootdir = "imdb/word_data/all"
    authors = get_author_names_all_authors(os.path.join(data_rootdir, "all_authors.txt"))
    test_fold = 0
    authors = authors[:authors_count]
    # print(authors)
    print("size, test_fold: ", authors_count, test_fold)
    X_train, Y_train, X_test, Y_test = get_authors_train_test(data_rootdir, authors, val_size=0,
                                                              train_ratio=train_ratio)

    return X_train, Y_train, X_test, Y_test


def get_blogs_data_for_nns(min_doc=None):
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for i in range(1, authors_count+1):
        print(i)
        doc_num_train = len(list(filter(lambda x: '_' not in str(x), os.listdir('blogs/'+str(authors_count)+'/train/author'+str(i)))))
        if min_doc is not None:
            doc_num_train = min(min_doc, doc_num_train)
        for j in range(1, int(doc_num_train + 1)):
            with open("blogs/"+str(authors_count)+"/train/author" + str(i) + "/" + str(j) + ".txt", "r", encoding='latin1') as f:
                document = f.read()
                document = document[:-1]
                X_train.append(document)
                Y_train.append([i - 1])
        doc_num_test = len(list(filter(lambda x: '_' not in str(x), os.listdir('blogs/'+str(authors_count)+'/test/author'+str(i)))))
        for j in range(1, int(doc_num_test + 1)):
            with open("blogs/"+str(authors_count)+"/test/author" + str(i) + "/" + str(j) + ".txt", "r", encoding='latin1') as f:
                document = f.read()
                document = document[:-1]
                X_test.append(document)
                Y_test.append([i - 1])

    Y_train_word = np_utils.to_categorical(Y_train)
    Y_test_word = np_utils.to_categorical(Y_test)
    return X_train, Y_train_word, X_test, Y_test_word


def train_model_with_params():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    models_results_opdir = os.path.join(results_opdir, timestamp)
    print(models_results_opdir)
    utils.create_opdir(models_results_opdir)

    # 加载CCAT数据集
    if dataset == 'CCAT' or dataset == 'imdb':
        X_train, Y_train_word, X_test, Y_test_word = get_data_for_nns()

    # 加载blogs数据集
    elif dataset == 'blogs':
        X_train, Y_train_word, X_test, Y_test_word = get_blogs_data_for_nns(min_doc=10)

    print("load end...")
    no_of_authors = args.num_authors

    embedding_dims = 300
    filter_tuple = [[500, 3], [500, 4], [500, 5]]
    filter_tuple1 = [[50, 3], [50, 4], [50, 5]]
    dropout = 0.25
    activation_value = "relu"
    nb_epochs = 300
    batch_size = 4
    learning_rate = 4 * 1e-4
    single_feats = Features(args.type)
    single_feats.fit(X_train)
    print('word')
    X_train_single, X_test_single, single_to_idx = single_feats.transform(X_train), single_feats.transform(
        X_test), single_feats.feature_map
    print('syntax')

    if dataset == 'CCAT' or dataset == 'imdb':
        X_train, Y_train1, X_test, Y_test1, dic, length, label_dic = get_CCAT_train_and_test(dataset, authors_count)
    elif dataset == 'blogs':
        X_train, Y_train1, X_test, Y_test1, dic, length, label_dic = get_blogs_train_and_test(authors_count, min_doc=10)

    X_train_single_syntax = transform(X_train, dic, length)
    X_test_single_syntax = transform(X_test, dic, length)

    print("start train......")
    train_and_test(X_train_single, X_train_single_syntax, Y_train_word, X_test_single, X_test_single_syntax,
                   Y_test_word, nb_epochs, batch_size, learning_rate, no_of_authors, 300, 60, single_to_idx,
                   filter_tuple, filter_tuple1, dropout, activation_value, length, label_dic, timestamp,
                   models_results_opdir)


def sava_data(X, Y, flag):
    for i in range(len(X)):
        with open(args.datasets + "/word_data10_/" + flag + "/author" + str(Y[i] + 1) + "/" + str(i) + ".txt",
                  "a+") as f:
            f.write(X[i])
            f.write("\n")


if __name__ == "__main__":
    train_model_with_params()
