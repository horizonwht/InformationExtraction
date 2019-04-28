#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 19-4-25 上午11:28
# @Author : wanghuiting
# @File   : utils.py
import pickle
from config.config import P_DATA_CLASSIFY_PKL
import numpy as np
import torch as th
from torch.autograd import Variable
from sklearn.metrics import recall_score, precision_score, f1_score


def save_pickle_data(data, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)


def load_pickle_data(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data

def load_train_test_data():
    data = load_pickle_data(P_DATA_CLASSIFY_PKL)
    train_x = data['train_x']
    train_pos = data['train_x_pos']
    train_y = data['train_y']
    test_x = data['test_x']
    test_pos = data['test_x_pos']
    test_y = data['test_y']
    return train_x,train_pos,train_y,test_x,test_pos,test_y

def generate_batch_data(x, x2, y, batch_size, max_len,use_cuda):
    """
    生成batch数据
    :param x:
    :param y:
    :param batch_size:
    :param max_len:
    :return:
    """
    data = []
    # total_batch = len(x) // batch_size
    # if len(x) % batch_size > 0:
    #     total_batch += 1
    total_batch = 100

    for ii in range(total_batch):
        start, end = ii * batch_size, (ii + 1) * batch_size
        end = min(end, len(x))
        batch_y = np.array(y[start:end], dtype=np.int32)
        batch_x = np.zeros((end - start, max_len), dtype=np.int32)
        batch_x2 = np.zeros((end - start, max_len), dtype=np.int32)
        seq_len = np.zeros((end - start))
        for kk, sentence in enumerate(x[start:end]):
            seq_l = len(sentence)
            batch_x[kk, :seq_l] = np.array(sentence)
            seq_len[kk] = seq_l
        for kk2, sentence2 in enumerate(x2[start:end]):
            seq_l = len(sentence2)
            batch_x2[kk2, :seq_l] = np.array(sentence2)
        batch_x = th.from_numpy(batch_x)
        batch_x2 = th.from_numpy(batch_x2)
        batch_y = th.from_numpy(batch_y)
        if use_cuda:
            batch_x, batch_x2, batch_y = batch_x.cuda(), batch_x2.cuda(), batch_y.cuda()
        batch_x, batch_x2, batch_y = Variable(batch_x).long(), Variable(batch_x2).long(), Variable(batch_y).long()
        # yield batch_x, batch_x2, batch_y
        data.append([batch_x, batch_x2, batch_y])
    return data


def print_info(epoch, train_loss, dev_loss, y, pre_y):
    loss = round(float(np.mean(train_loss)), 3)
    val_loss = round(float(np.mean(dev_loss)), 3)
    f1 = round(f1_score(y, pre_y), 4)
    recall = round(recall_score(y, pre_y), 4)
    precision = round(precision_score(y, pre_y), 4)
    print("epoch\t{}\ttrain_loss\t{}\tdev_loss\t{}\t".format(epoch, loss, val_loss))
    print("precision\t{}\trecall\t{}\tf1\t{}\n\n".format(precision, recall, f1))
    return f1
