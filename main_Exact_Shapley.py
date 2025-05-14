# sourcery skip: identity-comprehension
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from datetime import datetime
from time import strftime
import time
from tqdm import tqdm
import json

from client_split_sample.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from client_split_sample.Dirichlet_split_datasets import split_noniid
from client_split_sample.sampling_by_proportion import sample_by_proportion
from models.Update import LocalUpdate
from utils.weight_cal import para_diff_cal, float_mulpty_OrderedDict, normal_test
from models.Nets import MLP, CNNMnist, CNNCifar, CharLSTM
from models.qFed import weight_agg, gradient_agg
from models.test import test_img
from utils.dataset import ShakeSpeare
from utils.utils import exp_details, worst_fraction, best_fraction, adjust_quality
from utils.shapley import all_subsets, list2str, cal_shapley, shapley_rank


def load_gradient(path):
    """加载梯度"""
    gradient = torch.load(path, weights_only=False)
    return gradient

if __name__ == '__main__':
    args = args_parser()
    args.method = "exact_shapley"
    exp_details(args)  # 打印超参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu') # 使用cpu还是gpu 赋值args.device

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)  #训练集
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)  #测试集
        # y_train = np.array(dataset_train.targets)
        # # sample users
        # if args.iid:
        #     print("iid")
        #     dict_users = mnist_iid(dataset_train, args.num_users)  # 为用户分配iid数据
        # else:
        #     if args.dirichlet != 0:
        #         print("non-iid->dirichlet")
        #         labels_train = np.array(dataset_train.targets)
        #         dict_users = split_noniid(labels_train, args.dirichlet, args.num_users)
        #     else:
        #         print("non-iid->shard")
        #         dict_users = mnist_noniid(dataset_train, args.num_users)  # 否则为用户分配non-iid数据
        
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)  #训练集
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar)  #测试集
        # if args.iid:
        #     dict_users = cifar_iid(dataset_train, args.num_users)  # 为用户分配iid数据
        # else:
        #     if args.dirichlet != 0:
        #         print("dirichlet")
        #         labels_train = np.array(dataset_train.targets)
        #         dict_users = split_noniid(labels_train, args.dirichlet, args.num_users)
        #     else:
        #         print("shard")
        #         dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'shakespeare':
        # dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        # dict_users = dataset_train.get_client_dic()
        # args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    
    with open('./setup/dict_users.json', 'r') as f:
        read_dict_users = json.load(f)
    for key in read_dict_users.keys():
        print(key, len(read_dict_users[key]))
    
    if args.dataset == 'mnist':
        img_size = [1,28,28]
    elif args.dataset == 'cifar':
        img_size = [3,32,32]

    # 恢复模型状态
    if args.dataset == 'mnist':
        img_size = [1,28,28]
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.dataset == 'shakespeare':
        net_glob = CharLSTM().to(args.device)

    # 所有子集
    iterable = [i for i in range(args.num_users)]
    powerset = all_subsets(iterable)    # e.g. [[],[0],[1],[0.1]]
    # print(powerset)
    
    total_shapley_dict = {}
    for i in range(args.num_users):
        total_shapley_dict[i] = 0

    start_time = time.time()

    gradient_dir = './gradient_cache'
    
    for iter in range(args.epochs):
        print("Evaluation round: {}".format(iter))
        
        w_glob = load_gradient(f'./gradient_cache/global_model/model_epoch_{iter}.pth')
        net_glob.load_state_dict(w_glob)

        # 初始化全部子模型
        submodel_dict = {}
        # submodel_name_list = []
        for subset in powerset:
            # print(subset)
            # tuple_subset = tuple(subset)
            str_subset = list2str(subset)
            # print(str_subset)
            # submodel_name_list.append(str_subset)
            submodel_dict[str_subset] = copy.deepcopy(net_glob)
            submodel_dict[str_subset].to(args.device)
            # submodel_dict[str_subset].train()
        # print(submodel_name_list)
        # print(submodel_dict)
        
        w_glob_shapley = copy.deepcopy(w_glob)

        accuracy_dict = {}
        # print(len(w_locals))
        for set in tqdm(powerset):
            # print(set)
            if not set:
                ### 空集合就直接测试原始模型的性能
                test_acc, test_loss = test_img(submodel_dict[list2str(set)], dataset_test, args, "test_dataset")
                # print("not set")
            else:    
                # print(set)
                # 聚合权重计算
                ldv = [len(read_dict_users[str(cid)]) for cid in set]
                # print(ldv)
                tdv = sum(ldv)
                # print(tdv)
                agg_weights = [i / tdv for i in ldv]
                # print(agg_weights)
                # 从文件加载梯度
                agg_parameter = []
                for cid in set:
                    grad_path = os.path.join(gradient_dir, f'epoch_{iter}_client_{cid}.pt')
                    agg_parameter.append(load_gradient(grad_path))
                # print(agg_parameter)
                global_parameter = gradient_agg(agg_parameter, agg_weights, w_glob_shapley)
                submodel_dict[list2str(set)].load_state_dict(global_parameter)
                test_acc, test_loss = test_img(submodel_dict[list2str(set)], dataset_test, args, "test_dataset")
            accuracy_dict[list2str(set)] = test_acc
            
        # print(accuracy_dict)
        shapley_dict = cal_shapley(accuracy_dict, args.num_users)
        # # 边际贡献写入txt文件
        # with open('./save/Margin_contribution_exactshapley.txt', 'a') as f:
        #     f.write('Training round:' + str(iter) + '\n')
        #     for i in accuracy_dict:
        #         f.write('Client set ' + str(i) + '-> Margin contribution: ' + str(float(accuracy_dict[i])) + '\n')
        # SV写入txt文件
        with open('./save/Exact_Shapley.txt', 'a') as f:
            f.write('Training round:' + str(iter) + '\n')
            for i in shapley_dict:
                f.write('Client' + str(i) + '-> Shapley value: ' + str(float(shapley_dict[i])) + '\n')
        for i in range(args.num_users):
            total_shapley_dict[i] += shapley_dict[i]
        print(shapley_dict)
        """
        Shapley Calculation
        """
    print(total_shapley_dict)
    with open('./save/Exact_Shapley.txt', 'a') as f:
        f.write('\n\n')
        for key, value in total_shapley_dict.items():
            f.write(f"{key}: {value}" + '\t')
    print(shapley_rank(total_shapley_dict))
    with open('./save/Exact_Shapley.txt', 'a') as f:
        f.write('\n')
        f.write('Shapley rank: ' + str(shapley_rank(total_shapley_dict)))

    end_time =  time.time()
    execution_time = end_time - start_time
    print("程序运行时间为：", execution_time, "秒")
    with open('./save/Exact_Shapley.txt', 'a') as f:
        f.write('Running_time: ' + str(execution_time) + 's'+'\n')

    with open('./shapley_result/running_time.txt', 'a') as f:
        f.write('Exact_shapley: client number_' + str(args.num_users) + ' running_time: ' + str(execution_time) + 's'+'\n')