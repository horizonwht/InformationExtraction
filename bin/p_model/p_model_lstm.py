#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 19-4-23 下午1:54
# @Author : wanghuiting
# @File   : p_model_lstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class P_Model_LSTM(nn.Module):
    def __init__(self, hidden_dim,class_dim, word_vocab_size, word_dim, postag_vocab_size, postag_dim):
        super(P_Model_LSTM, self).__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_dim)
        self.postag_embedding = nn.Embedding(postag_vocab_size, postag_dim)
        self.fc1 = nn.Linear(word_dim, hidden_dim)
        self.fc2 = nn.Linear(postag_dim, hidden_dim)
        self.fc3 = nn.Linear(4*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, class_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, word, postag):
        """
        :param word:
        :param postag:
        :return:
        """
        word = self.word_embedding(word)
        postag = self.postag_embedding(postag)
        hidden_0 = self.fc1(word) + self.fc2(postag)
        x, _ = self.lstm1(hidden_0)
        x = self.pool_layer(x)
        output = torch.tanh(self.fc3(x))
        output = self.fc4(output)
        return output

    @staticmethod
    def pool_layer(x):
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        return torch.cat((p1, p2), 1)

    def class_loss_layer(self,log_out, y_batch):
        """
        分类损失函数
        :param log_out:
        :param y_batch:
        :return:
        """
        batch_size, class_num = log_out.size()
        y = torch.zeros(batch_size, class_num).scatter_(1,y_batch, 1)
        loss = -torch.sum(y * log_out,dim=1)
        loss = torch.mean(loss)
        return loss
