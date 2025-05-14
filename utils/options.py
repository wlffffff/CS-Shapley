#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()  # 创建一个解析器（Argument Parser()对象）
    # federated arguments     # 添加参数 调用add_argument()方法添加参数
    parser.add_argument('--epochs', type=int, default=3, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--test_dataset_ratio', type=float, default=1, help="the ratio of test_dataset")
    parser.add_argument('--bs_test', type=int, default=512, help="test batch size")
    parser.add_argument('--bs_train', type=int, default=128, help="train batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--dirichlet', type=float, default=0.0, help="dirichlet ratio to split dataset")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    
    # noise arguments
    parser.add_argument('--add_noise', type=int, default=-1, help="add noise or not")
    parser.add_argument('--LID_k', type=int, default=20, help="lid")
    parser.add_argument('--level_n_system', type=float, default=0.3, help="fraction of noisy clients")
    parser.add_argument('--level_n_lowerb', type=float, default=0.8, help="lower bound of noise level")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--iid', type=int, default=0,help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--sample_by_proportion', default=False, action='store_true', help='sample by number of local datasets')

    parser.add_argument('--mc_iteration', type=int, default=30, help='parameter of tmc_shapley')
    parser.add_argument('--tolerance', type=float, default=0.0005, help='parameter of tmc_shapley')

    parser.add_argument('--B_L', type=float, default=0.0, help='parameter of delta_shapley')
    parser.add_argument('--B_U', type=float, default=0.4, help='parameter of delta_shapley')

    parser.add_argument('--para_H', type=float, default=50, help='parameter of mab_shapley "H"')
    parser.add_argument('--para_K', type=float, default=0.5, help='parameter of mab_shapley "K"')
    parser.add_argument('--iteration', type=int, default=100, help='parameter of mab_mc_shapley')

    parser.add_argument('--num', type=int, default=3, help='select clients based on the SV order')

    # CS-Shapley
    parser.add_argument('--selection_ratio', type=float, default=0.9, help='parameter of cs_shapley "selection client ratio"')
    parser.add_argument('--overlook_ratio', type=float, default=0.8, help='parameter of cs_shapley "overlook pumation ratio"')

    # CGSV
    parser.add_argument('--Gamma', type=float, default=0.5, help='parameter of CGSV "Gamma"')
    parser.add_argument('--alpha', type=float, default=0.95, help='parameter of CGSV "alpha"')

    # Quality
    parser.add_argument('--data_quality_mode', type=str, default='clean', choices=['clean', 'noise', 'blur', 'count', 'mask', 'flip'],
                      help='Data quality adjustment mode')
    parser.add_argument('--visualize', action='store_true', help='Enable data visualization')
    # parser.add_argument('--quality_level', type=int, default=10,help='Quality degradation level (1-10)')
    
    args = parser.parse_args()  # 解析参数 使用parse_args()解析添加的参数
    return args
