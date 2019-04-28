#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 19-4-25 下午5:08
# @Author : wanghuiting
# @File   : p_model_textcnn.py

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, channel,class_dim,word_vocab_size, word_dim,postag_vocab_size,postag_dim):
        super(TextCNN, self).__init__()
        # 需要将事先训练好的词向量载入
        self.word_embeding = nn.Embedding(word_vocab_size, word_dim)
        self.pos_embeding = nn.Embedding(postag_vocab_size, postag_dim)
        self.conv1 = nn.Conv2d(1, channel, kernel_size=(2, word_dim+postag_dim))
        self.conv2 = nn.Conv2d(1, channel, kernel_size=(3, word_dim+postag_dim))
        self.conv3 = nn.Conv2d(1, channel, kernel_size=(4, word_dim+postag_dim))
        self.conv4 = nn.Conv2d(1, channel, kernel_size=(5, word_dim+postag_dim))
        self.fc1 = nn.Linear(4 * channel, channel)
        self.fc2 = nn.Linear(channel, class_dim)

    def forward(self, *input):
        """
        :param input:
        :return:
        """
        word, pos = input[0], input[1]
        word = self.word_embeding(word)
        pos = self.pos_embeding(pos)
        x = torch.cat((word,pos),dim=-1)
        x = x.unsqueeze(dim=1)  # 添加维度
        conv1 = self.cnn_layer(self.conv1(x))
        conv2 = self.cnn_layer(self.conv2(x))
        conv3 = self.cnn_layer(self.conv3(x))
        conv4 = self.cnn_layer(self.conv4(x))
        output = torch.cat((conv1, conv2, conv3, conv4), dim=-1)
        output = torch.tanh(self.fc1(output))
        output = self.fc2(output)
        return output

    def cnn_layer(self, inputs):
        output = F.relu(inputs)
        output = F.max_pool2d(output, kernel_size=(output.size(2), 1))
        output = output.view(output.size(0), output.size(1))
        return output

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

    def loss(self,test_y, predict, num_class):
        loss = 0
        predict = F.sigmoid(predict)
        for ii in range(num_class):
            loss += criterion(predict[:, ii], test_y[:, ii])
            acc = (predict[:, ii].ge(0.5) == test_y[:, ii]).sum() / predict.shape[0]


