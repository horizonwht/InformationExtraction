#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time   : 19-4-25 上午9:52
# @Author : wanghuiting
# @File   : p_preprocessor.py
from config.config import ConfigDict, VOCAB_PATH, P_DATA_CLASSIFY_PKL
import os
import codecs
import json
from config.utils import save_pickle_data


class Prepeocessor(object):
    def __init__(self):
        self.wordemb_dict_path = ConfigDict['word_idx_path']
        self.postag_dict_path = ConfigDict['postag_dict_path']
        self.label_dict_path = ConfigDict['label_dict_path']
        self.train_data_list_path = ConfigDict['train_data_path']
        self.test_data_list_path = ConfigDict['test_data_path']
        self._UNK_IDX = 0
        self.p_map_eng_dict = {}
        self.feature_dict = {}
        # load dictionary
        self._dict_path_dict = {'wordemb_dict': self.wordemb_dict_path,
                                'postag_dict': self.postag_dict_path,
                                'label_dict': self.label_dict_path}
        # check if the file exists
        for input_dict in [self.wordemb_dict_path, self.postag_dict_path, self.label_dict_path,
                           self.train_data_list_path, self.test_data_list_path]:
            if not os.path.exists(input_dict):
                raise ValueError("%s not found." % (input_dict))
                return

    def load_label_dict(self, dict_name):
        """load label dict from file"""
        label_dict = {}
        with codecs.open(dict_name, 'r', 'utf-8') as fr:
            for idx, line in enumerate(fr):
                p, p_eng = line.strip().split('\t')
                label_dict[p_eng] = idx
                self.p_map_eng_dict[p] = p_eng
        return label_dict

    def load_dict_from_file(self, dict_name, bias=0):
        """
        Load vocabulary from file.
        """
        dict_result = {}
        with codecs.open(dict_name, 'r', 'utf-8') as f_dict:
            for idx, line in enumerate(f_dict):
                line = line.strip()
                dict_result[line] = idx + bias
        return dict_result

    def get_reverse_dict(self, dict_name):
        dict_reverse = {}
        for key, value in self.feature_dict[dict_name].items():
            dict_reverse[value] = key
        return dict_reverse

    def reverse_p_eng(self, dic):
        dict_reverse = {}
        for key, value in dic.items():
            dict_reverse[value] = key
        return dict_reverse

    def load_data(self, data_path, need_input=False, need_label=True):
        """
        读取训练数据
        :return:
        """
        result = []
        max_len = 0
        pos_max_len = 0
        if os.path.isdir(data_path):
            input_files = os.listdir(data_path)
            for data_file in input_files:
                data_file_path = os.path.join(data_path, data_file)

                for line in open(data_file_path.strip()):
                    sample_result = self.get_feed_iterator(line.strip(), need_input, need_label)
                    if sample_result is None:
                        continue
                    max_len = max(max_len, len(sample_result[0]))
                    pos_max_len = max(pos_max_len,len(sample_result[1]))
                    result.append(sample_result)
        elif os.path.isfile(data_path):
            for line in open(data_path.strip()):
                sample_result = self.get_feed_iterator(line.strip(), need_input, need_label)
                if sample_result is None:
                    continue
                max_len = max(max_len, len(sample_result[0]))
                pos_max_len = max(pos_max_len, len(sample_result[1]))
                result.append(sample_result)

        return result, max_len,pos_max_len

    def cal_mark_slot(self, spo_list):
        """
        Calculate the value of the label
        """
        mark_list = [0] * len(self.feature_dict['label_dict'])
        for spo in spo_list:
            predicate = spo['predicate']
            p_idx = self.feature_dict['label_dict'][self.p_map_eng_dict[predicate]]
            mark_list[p_idx] = 1
        return mark_list

    def get_feed_iterator(self, line, need_input=False, need_label=True):
        """

        :param line:
        :param need_input:
        :param need_label:
        :return:
        """
        dic = json.loads(line)
        sentence_term_list = [item['word'] for item in dic['postag']]
        sentence_pos_list = [item['pos'] for item in dic['postag']]
        # 词id
        sentence_emb_slot = [self.feature_dict['wordemb_dict'].get(w, self._UNK_IDX) \
                             for w in sentence_term_list]
        # 词性id (n,f,s,v)
        sentence_pos_slot = [self.feature_dict['postag_dict'].get(pos, self._UNK_IDX) \
                             for pos in sentence_pos_list]
        # label_slot --> [0,1,0,0] 长度为49 (p_eng的长度)
        if 'spo_list' not in dic:
            label_slot = [0] * len(self.feature_dict['label_dict'])
        else:
            label_slot = self.cal_mark_slot(dic['spo_list'])
        # verify that the feature is valid
        if len(sentence_emb_slot) == 0 or len(sentence_pos_slot) == 0 \
                or len(label_slot) == 0:
            return None
        feature_slot = [sentence_emb_slot, sentence_pos_slot]
        # print(feature_slot)
        input_fields = json.dumps(dic, ensure_ascii=False)
        output_slot = feature_slot
        if need_input:
            output_slot = [input_fields] + output_slot
        if need_label:
            output_slot = output_slot + [label_slot]
        return output_slot

    def process(self):
        """
        :return:
        """
        self.feature_dict['postag_dict'] = \
            self.load_dict_from_file(self._dict_path_dict['postag_dict'])
        # print(self._dict_path_dict['wordemb_dict'])
        self.feature_dict['wordemb_dict'] = \
            self.load_dict_from_file(self._dict_path_dict['wordemb_dict'])
        # label_dict --> p_eng
        self.feature_dict['label_dict'] = \
            self.load_label_dict(self._dict_path_dict['label_dict'])
        # print(self._feature_dict['label_dict'])
        self.reverse_dict = {name: self.get_reverse_dict(name) for name in
                             self._dict_path_dict.keys()}
        self.reverse_dict['eng_map_p_dict'] = self.reverse_p_eng(self.p_map_eng_dict)


        train_data, train_max_len,train_pos_max_len = self.load_data(self.train_data_list_path)
        test_data, test_max_len,test_pos_max_len = self.load_data(self.test_data_list_path)
        max_len = max(train_max_len, test_max_len)
        self.feature_dict['max_len'] = max_len
        train_x, train_x_pos, train_p = zip(*[[row[0], row[1], row[2]] for row in train_data])
        test_x, test_x_pos, test_p = zip(*[[row[0], row[1], row[2]] for row in test_data])
        print("数据量\t{}\t训练集\t{}\t测试集\t{}".format(len(train_x) + len(test_x), len(train_x), len(test_x)))
        data = {
            "train_x": train_x,
            "train_x_pos": train_x_pos,
            "train_y": train_p,
            "test_x": test_x,
            "test_x_pos": test_x_pos,
            "test_y": test_p,
        }

        save_pickle_data(self.feature_dict, VOCAB_PATH)
        save_pickle_data(data, P_DATA_CLASSIFY_PKL)
