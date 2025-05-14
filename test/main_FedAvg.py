#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from datetime import datetime
from time import strftime
import os
import time
import json

from client_split_sample.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CharLSTM
from models.Fed import FedWeightAvg, weight_agg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from utils.utils import exp_details, worst_fraction, best_fraction


if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.method = "fedavg_non-shapley"
    exp_details(args)  # 打印超参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # # sample users
        # if args.iid:
        #     dict_users = mnist_iid(dataset_train, args.num_users)
        # else:
        #     dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        # if args.iid:
        #     dict_users = cifar_iid(dataset_train, args.num_users)
        # else:
        #     dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')

    # # print(dict_users)
    # new_dict_users = {}
    # for key,value in dict_users.items():
    #     new_dict_users[key] = value.tolist()
    # # print(new_dict_users)
    # with open('./setup/dict_users.json', 'w') as f:
    #     json.dump(new_dict_users, f)
    
    with open('./setup/dict_users.json', 'r') as f:
        read_dict_users = json.load(f)
    # print(read_dict_users)
    # print(read_dict_users['3'])

    if args.dataset == 'mnist':
        img_size = [1,28,28]
    elif args.dataset == 'cifar':
        img_size = [3,32,32]
    # print(img_size)

    # # build model
    # if args.model == 'cnn' and args.dataset == 'cifar':
    #     net_glob = CNNCifar(args=args).to(args.device)
    # elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
    #     net_glob = CNNMnist(args=args).to(args.device)
    # elif args.dataset == 'femnist' and args.model == 'cnn':
    #     net_glob = CNNFemnist(args=args).to(args.device)
    # elif args.dataset == 'shakespeare' and args.model == 'lstm':
    #     net_glob = CharLSTM().to(args.device)
    # elif args.model == 'mlp':
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #     net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    # else:
    #     exit('Error: unrecognized model')
    # print(net_glob)
    # print('Number of clients: {:3d}'.format(args.num_users))

    # torch.save(net_glob, './setup/model.pth')

    net_glob = torch.load('./setup/model.pth')

    net_glob.train()
    w_glob = net_glob.state_dict()

    # training
    loss_train = []

    start_time = time.time()

    for iter in range(args.epochs):
        loss_locals = []  # 对于每一个epoch，初始化worker的损失
        w_locals = []  # 存储客户端本地权重

        net_glob.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs_users)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=read_dict_users[str(idx)]) # 对每个worker进行本地更新
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device)) # 本地训练的weight和loss  ##第5行完成

            local_data_volume = [len(read_dict_users[str(cid)]) for cid in range(len(idxs_users))]
            total_data_volume = sum(local_data_volume)
            weights = [l_d_v / total_data_volume for l_d_v in local_data_volume]

            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        w_glob = weight_agg(w_locals, weights)
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fedavg_{}_{}_{}_C{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, datetime.now().strftime("%H_%M_%S")))

    # testing  测试集上进行测试
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args, "train_dataset")
    acc_test, loss_test = test_img(net_glob, dataset_test, args, "test_dataset")

    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    end_time =  time.time()
    execution_time = end_time - start_time
    print("程序运行时间为：", execution_time, "秒")