#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 19-4-24 下午3:43
# @Author : wanghuiting
# @File   : run_p_model.py

import torch
from bin.p_model.p_model_lstm import P_Model_LSTM
from bin.p_model.p_model_textcnn import TextCNN
from bin.p_model.p_preprocessor import Prepeocessor
from config.config import ConfigDict, VOCAB_PATH, MODEL_PATH
from config.utils import load_pickle_data, load_train_test_data, generate_batch_data, print_info
from torch import optim
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import classification_report
tqdm.pandas(desc="my bar!")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def preprocess():
    """
    预处理
    :return:
    """
    model = Prepeocessor()
    model.process()


def run():
    use_cuda = torch.cuda.is_available()
    vocab = load_pickle_data(VOCAB_PATH)
    word_vocab_size = len(vocab['wordemb_dict'])
    postag_vocab_size = len(vocab['postag_dict'])
    class_dim = len(vocab['label_dict'])
    max_len = vocab.get('max_len')
    train_x, train_pos, train_y, test_x, test_pos, test_y = load_train_test_data()
    # model = P_Model_LSTM(ConfigDict['hidden_dim'], class_dim, word_vocab_size, ConfigDict['word_dim'],
    #                      postag_vocab_size, ConfigDict['postag_dim'])
    model = TextCNN(ConfigDict['hidden_dim'], class_dim,word_vocab_size, ConfigDict['word_dim'],
                         postag_vocab_size, ConfigDict['postag_dim'])
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    if use_cuda:
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    max_f1 = 0
    for epoch in range(ConfigDict['epoch']):
        train_data = generate_batch_data(train_x, train_pos, train_y,ConfigDict['batch_size'], max_len, use_cuda)
        for num_iter,batch_data in enumerate(tqdm(train_data,desc='训练')):
            x_batch, pos_batch, y_batch = batch_data
            optimizer.zero_grad()
            output = model(x_batch, pos_batch)
            # loss = fs.nll_loss(F.log_softmax(output, dim=-1), y_batch)
            loss = model.class_loss_layer(F.log_softmax(output, dim=-1), y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        test_loss = 0
        predicts = torch.Tensor()
        test_data = generate_batch_data(test_x, test_pos, test_y,ConfigDict['batch_size'], max_len, use_cuda)
        for num_iter,batch_data in enumerate(tqdm(test_data,desc='测试')):
            x_batch, pos_batch, y_batch = batch_data
            output = model(x_batch, pos_batch)
            # loss = fs.nll_loss(F.log_softmax(output, dim=-1), y_batch)
            test_loss += float(model.class_loss_layer(F.log_softmax(output, dim=-1), y_batch))
            pred = output.float().cpu()
            predicts = torch.cat((predicts, pred))
        predicts = predicts.detach().numpy()

        predicts = np.argmax(predicts, axis=-1)
        print(test_y)
        print_info = classification_report(test_y, predicts)
        f1 = float(print_info.strip().split('\n')[-1].split()[-2])
        if max_f1 < f1:
            max_f1 = f1
            torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    # preprocess()
    run()
