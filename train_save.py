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
from torchvision.models import resnet18
import torch
from datetime import datetime
from time import strftime
import os
import time
import json
import torch.nn as nn

from client_split_sample.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from client_split_sample.Dirichlet_split_datasets import split_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CharLSTM
from models.Fed import FedWeightAvg, weight_agg
from models.test import test_img
from models.qFed import weight_agg, gradient_agg
from utils.dataset import FEMNIST, ShakeSpeare
from utils.utils import exp_details, worst_fraction, best_fraction, add_noise, adjust_quality


def save_gradient(gradient, path):
    """
    保存梯度到文件
    :param gradient: 梯度
    :param path: 保存路径
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(gradient, path)

if __name__ == '__main__':
    args = args_parser()
    args.method = "train_save_model"
    exp_details(args)  # 打印超参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)  #训练集
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)  #测试集
        # sample users
        if args.iid:
            print("iid")
            dict_users = mnist_iid(dataset_train, args.num_users)  # 为用户分配iid数据
        else:
            if args.dirichlet != 0:
                print("non-iid->dirichlet")
                labels_train = np.array(dataset_train.targets)
                dict_users = split_noniid(labels_train, args.dirichlet, args.num_users)
            else:
                print("non-iid->shard")
                dict_users = mnist_noniid(dataset_train, args.num_users)  # 否则为用户分配non-iid数据
        
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    # 
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)  #训练集
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)  #测试集
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)  # 为用户分配iid数据
        else:
            if args.dirichlet != 0:
                print("dirichlet")
                labels_train = np.array(dataset_train.targets)
                dict_users = split_noniid(labels_train, args.dirichlet, args.num_users)
            else:
                print("shard")
                dict_users = cifar_noniid(dataset_train, args.num_users)
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
        

    new_dict_users = {}
    for key,value in dict_users.items():
        new_dict_users[key] = [int(x) for x in value]
    # 打印每个用户的数据量
    for key in new_dict_users.keys():
        print(key, len(new_dict_users[key]))

    with open('./setup/dict_users.json', 'w') as f:
        f.truncate(0)
        json.dump(new_dict_users, f)
    

    if args.dataset == 'mnist':
        img_size = [1,28,28]
    elif args.dataset == 'cifar':
        img_size = [3,32,32]
    # print(img_size)

    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'resnet' and args.dataset == 'cifar':
        resnet18_model = resnet18(pretrained=False, num_classes=args.num_classes)
        resnet18_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet18_model.maxpool = nn.Identity()
        net_glob = resnet18_model.to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # torch.save(net_glob, './setup/model.pth')

    # print("初始化完成！")

    # loaded_model = torch.load('./setup/model.pth')

    w_glob = net_glob.state_dict()
    torch.save(net_glob.state_dict(), f'./gradient_cache/global_model/model_epoch_0.pth')

    # 创建梯度保存目录
    gradient_dir = './gradient_cache'
    os.makedirs(gradient_dir, exist_ok=True)
    
    for iter in range(args.epochs):
        loss_locals = []  # 对于每一个epoch，初始化worker的损失
        gradient_locals = []  # 存储客户端本地权重

        net_glob.train()
        idxs_users = [i for i in range(args.num_users)]
        
        for idx in idxs_users:  # 对于选取的m个worker
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[int(idx)]) # 对每个worker进行本地更新
            update, loss = local.gradient_train(net=copy.deepcopy(net_glob).to(args.device)) # 本地训练的weight和loss  ##第5行完成
            # 保存梯度到文件
            grad_path = os.path.join(gradient_dir, f'epoch_{iter}_client_{idx}.pt')
            save_gradient(update, grad_path)
 
            gradient_locals.append(copy.deepcopy(update))
            loss_locals.append(copy.deepcopy(loss))

        local_data_volume = [len(dict_users[int(cid)]) for cid in idxs_users]
        total_data_volume = sum(local_data_volume)
        weights = [l_d_v / total_data_volume for l_d_v in local_data_volume]

        w_glob = gradient_agg(gradient_locals, weights, w_glob)
        net_glob.load_state_dict(w_glob)
        # print(gradient_locals)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        # loss_train_list.append(loss_avg)

        # acc
        net_glob.eval()
        
        acc_test, loss_test = test_img(net_glob, dataset_test, args, "test_dataset")
        print("###############")
        # acc_train, loss_train = test_img(net_glob, dataset_train, args, "train_dataset")
        # print(acc_test, acc_train, loss_train, loss_test)
        # print("Epoch {} Training accuracy: {:.2f}".format(iter, acc_train))
        print("Epoch {} Testing accuracy: {}".format(iter, acc_test))

        torch.save(net_glob.state_dict(), f'./gradient_cache/global_model/model_epoch_{iter+1}.pth')
        print("Training completed. Gradients saved to", gradient_dir)



        