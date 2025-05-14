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
from client_split_sample.Dirichlet_split_datasets import split_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CharLSTM
from models.Fed import FedWeightAvg, weight_agg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from utils.utils import exp_details, worst_fraction, best_fraction, add_noise, adjust_quality


def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# # 可视化时反归一化（针对CIFAR10的示例）
# def reverse_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
#     if isinstance(tensor, torch.Tensor):
#         tensor = tensor.clone()
#         for t, m, s in zip(tensor, mean, std):
#             t.mul_(s).add_(m)  # t = t*s + m
#         return tensor.permute(1, 2, 0).numpy()
#     else:  # numpy array
#         tensor = tensor.copy()
#         for c in range(tensor.shape[-1]):
#             tensor[..., c] = tensor[..., c] * std[c] + mean[c]
#         return tensor

def compare_datasets(original_dataset, modified_dataset, dict_users, args, num_samples=10):
    """
    对比原始数据集和修改后数据集的可视化
    :param original_dataset: 原始数据集
    :param modified_dataset: 修改后的数据集
    :param dict_users: 用户数据索引字典
    :param args: 命令行参数
    :param num_samples: 每个客户端显示的样本数
    """
    os.makedirs('./setup/visualization/comparison', exist_ok=True)
    
    # 选择前几个客户端进行可视化
    for client_id in range(args.num_users): 
        indices = dict_users[client_id]
        sample_indices = random.sample(indices, min(num_samples, len(indices)))
        
        plt.figure(figsize=(15, 4))
        plt.suptitle(f'Client {client_id} Data Quality Comparison ({args.data_quality_mode})')
        
        for i, idx in enumerate(sample_indices):
            # 获取原始和修改后的样本
            orig_img, orig_label = original_dataset[idx]
            mod_img, mod_label = modified_dataset[idx]

            # orig_img = reverse_normalize(orig_img)
            # mod_img = reverse_normalize(mod_img) if torch.is_tensor(mod_img) else np.array(mod_img)
            # print(mod_img)

            # 确保值在[0,1]范围内
            orig_img = np.clip(orig_img, 0, 1)
            mod_img = np.clip(mod_img, 0, 1) if isinstance(mod_img, np.ndarray) else mod_img

            orig_img = orig_img.permute(1, 2, 0).numpy() if isinstance(orig_img, torch.Tensor) else orig_img
            mod_img = mod_img.permute(1, 2, 0).numpy() if isinstance(mod_img, torch.Tensor) else mod_img
            
            # 原始数据
            plt.subplot(2, num_samples, i+1)
            plt.imshow(orig_img)
            plt.title(f"Original\nLabel: {orig_label}")
            plt.axis('off')
            
            # 修改后数据
            plt.subplot(2, num_samples, num_samples+i+1)
            plt.imshow(mod_img)
            plt.title(f"Modified\nLabel: {mod_label}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'./setup/visualization/comparison/client_{client_id}_comparison.png')
        plt.close()

if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.method = "setup"
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
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    # , transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

    # 调整客户端数据质量数量
    # # print("Original dataset:", dataset_train)
    # dict_users, adjusted_datasets = adjust_quality(dataset_train, dict_users, args)
    # # print("Modified dataset:", adjusted_datasets)
    
    # # 可视化对比
    # if args.visualize:
    #     compare_datasets(dataset_train, adjusted_datasets, dict_users, args)
        

    # 更新后的字典保存
    # print(dict_users)
    new_dict_users = {}
    for key,value in dict_users.items():
        new_dict_users[key] = [int(x) for x in value]

    # 打印每个用户的数据量
    data_volume = []
    for key in new_dict_users.keys():
        data_volume.append(len(new_dict_users[key]))
        # print(key, len(new_dict_users[key]))
    print(data_volume)
    # # print(new_dict_users)

    with open('./setup/dict_users.json', 'w') as f:
        f.truncate(0)
        json.dump(new_dict_users, f)
    

    if args.dataset == 'mnist':
        img_size = [1,28,28]
    elif args.dataset == 'cifar':
        img_size = [3,32,32]
    # print(img_size)

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # print(net_glob)
    # print('Number of clients: {:3d}'.format(args.num_users))

    torch.save(net_glob, './setup/model.pth')

    # loaded_model = torch.load('./setup/model.pth')



        