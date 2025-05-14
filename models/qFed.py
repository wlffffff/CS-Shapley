#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
from utils.weight_cal import weight_sum, para_diff_cal, delta_kt_sum, float_mulpty_OrderedDict, para_sum
import collections


def FedAvg(w):   # main中将w_locals赋给w，即worker计算出的权值
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # 对于每个参与的设备：
        for i in range(1, len(w)):  # 对本地更新进行聚合
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def gradient_agg(gradient_list, weight_list, origin_model):
    gradient_list_copy = copy.deepcopy(gradient_list)
    weight_list_copy = copy.deepcopy(weight_list)
    # for i, j in zip(gradient_list_copy, weight_list_copy):
    #     print(i)
    #     print(j)
    new_model_value = [float_mulpty_OrderedDict(j, i) for i, j in zip(gradient_list_copy, weight_list_copy)]
    # print(new_model_value)
    # new_model = dict(zip(new_model_key, new_model_value))
    new_model_value = weight_sum(new_model_value)
    # print(new_model_value)
    # print(collections.OrderedDict(new_model_value))
    return para_sum(collections.OrderedDict(new_model_value), origin_model)

def weight_agg(model_list, weight_list):
    new_model_key = model_list[0].keys()
    new_model_value = [float_mulpty_OrderedDict(j, i) for i, j in zip(model_list, weight_list)]
    # new_model = dict(zip(new_model_key, new_model_value))
    new_model_value = weight_sum(new_model_value)
    # print(new_model_value)
    # print(collections.OrderedDict(new_model_value))
    return collections.OrderedDict(new_model_value)


def q_FedAvg(global_model, delta_ks, hks):   # main中将w_locals赋给w，即worker计算出的权值
    keys = global_model.keys()
    
    sum_delta_kt = delta_kt_sum(delta_ks)
    sum_h_kt = np.sum(np.asarray(hks))
    values = [tensor_value/sum_h_kt for tensor_value in sum_delta_kt]
    global_model = [(x - y) for x, y in zip(global_model.values(), values)]

    new_model = dict(zip(keys, global_model))
    return new_model
