# sourcery skip: identity-comprehension
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
from torch.linalg import norm
import torch.nn.functional as F
from collections import OrderedDict
from typing import List, Dict

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


class CoAst:
    def __init__(self, num_clients: int, pruning_rate: float = 0.1, alpha: float = 0.02, k: int = 2):
        self.num_clients = num_clients
        self.pruning_rate = pruning_rate
        self.alpha = alpha
        self.k = k
        self.global_updates = []  # Stores global updates for cross-round valuation

    def prune_parameters(self, local_updates: List[Dict[str, torch.Tensor]], global_model: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        pruned_updates = []
        for delta in local_updates:
            pruned_delta = OrderedDict()
            for name, param in delta.items():
                abs_delta = torch.abs(param)
                threshold = torch.quantile(abs_delta.flatten(), 1 - self.pruning_rate)
                
                pruned_param = torch.where(
                    param > threshold,      
                    torch.ones_like(param) * self.alpha,
                    torch.where(
                        param < -threshold, 
                        -torch.ones_like(param) * self.alpha,
                        torch.zeros_like(param)
                    )
                )
                pruned_delta[name] = pruned_param + global_model[name]
            pruned_updates.append(pruned_delta)
        return pruned_updates

    def aggregate(self, pruned_updates: List[Dict[str, torch.Tensor]], global_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        global_update = OrderedDict()
        for name in global_model.keys():
            global_update[name] = torch.stack([update[name] for update in pruned_updates]).mean(dim=0)
        
        if len(self.global_updates) >= self.k:
            self.global_updates.pop(0)
        self.global_updates.append(global_update)
        
        # new_global_model = OrderedDict()
        # for name in global_model.keys():
        #     new_global_model[name] = global_model[name] + global_update[name]
        # return new_global_model
        return global_update

    def cross_round_valuation(self, local_models: List[Dict[str, torch.Tensor]]) -> List[float]:
        # print(len(self.global_updates))
        if len(self.global_updates) < self.k:
            return [1.0 / self.num_clients] * self.num_clients
        
        U_tk = OrderedDict()
        for name in self.global_updates[0].keys():
            U_tk[name] = torch.stack([update[name] for update in self.global_updates]).sum(dim=0)
        
        contributions = []
        for model in local_models:
            # print(model)
            score = 0.0
            for name in model.keys():
                theta = torch.sign(model[name].flatten())
                U = torch.sign(U_tk[name].flatten())
                score += torch.dot(theta, U).item()
            contributions.append(score)
        
        contributions = np.array(contributions)
        contributions = contributions / np.sum(np.abs(contributions))
        return contributions.tolist()


if __name__ == '__main__':
    args = args_parser()
    args.method = "CoAst"
    exp_details(args)  # 打印超参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu') # 使用cpu还是gpu 赋值args.device

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)  #训练集
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)  #测试集
        y_train = np.array(dataset_train.targets)
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
        trans_cifar = transforms.Compose([transforms.ToTensor()])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar)  #训练集
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
    
    with open('./setup/dict_users.json', 'r') as f:
        read_dict_users = json.load(f)

    """
    数据集处理
    """
    # 打印每个用户的数据量
    data_volume = []
    for key in read_dict_users.keys():
        data_volume.append(len(read_dict_users[key]))
        # print(key, len(new_dict_users[key]))
    print(data_volume)
    # # print(new_dict_users)
    
    # 调整客户端数据质量数量
    _, adjusted_datasets = adjust_quality(dataset_train, read_dict_users, args)
    """
    数据集处理
    """

    if args.dataset == 'mnist':
        img_size = [1,28,28]
    elif args.dataset == 'cifar':
        img_size = [3,32,32]

    # img_size = dataset_train[0][0].shape  # 图像的size

    # # build model
    # if args.model == 'cnn' and args.dataset == 'cifar':
    #     net_glob = CNNCifar(args=args).to(args.device)
    # elif args.model == 'cnn' and args.dataset == 'mnist':
    #     net_glob = CNNMnist(args=args).to(args.device)
    # elif args.dataset == 'shakespeare' and args.model == 'lstm':
    #     net_glob = CharLSTM().to(args.device)
    # elif args.model == 'mlp':
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #     net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    # else:
    #     exit('Error: unrecognized model')
    # # print(net_glob)

    net_glob = torch.load('./setup/model.pth')

    # with open('./save/Exact_Shapley.txt', 'a') as f:
    #     f.truncate(0)
    # with open('./save/Margin_contribution_exactshapley.txt', 'a') as f:
    #     f.truncate(0)

    net_glob.train()
    w_glob = net_glob.state_dict()
    
    # Initialize CoAst
    coast = CoAst(num_clients=args.num_users, pruning_rate=0.1, alpha=0.02, k=2)
    total_contributions = {i: 0 for i in range(args.num_users)}

    # training
    # loss_train_list = []

    # # 本地数据量
    # local_data_volume = [len(dict_users[cid]) for cid in range(args.num_users)]
    # total_data_volume = sum(local_data_volume)
    # print(local_data_volume)
    # print(total_data_volume)
    total_shapley_dict = {}
    for i in range(args.num_users):
        total_shapley_dict[i] = 0

    start_time = time.time()
    
    for iter in range(args.epochs):
        """
        Contribution Calculation
        """
        
        w_glob_coast= copy.deepcopy(w_glob)
        """
        Contribution Calculation
        """

        loss_locals = []  # 对于每一个epoch，初始化worker的损失
        gradient_locals = []  # 存储客户端本地权重
        local_models = []

        net_glob.train()
        # m = max(int(args.frac * args.num_users), 1)
        # print(m)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print(idxs_users)
        idxs_users = [i for i in range(args.num_users)]
        # SV由低到高排序iid [2, 6, 8, 5, 3, 1, 7, 4, 0, 9]
        # SV由低到高排序non-iid alpha = 1 [7, 8, 3, 5, 9, 2, 4, 6, 0, 1]
        # SV由低到高排序non-iid alpha = 0.5 [8, 3, 9, 4, 2, 1, 6, 0, 7, 5]
        # idxs_users = [8, 3, 9, 4, 2, 1, 6, 0, 7, 5]
        # idxs_users = idxs_users[-(args.num):]
        
        for idx in idxs_users:  # 对于选取的m个worker
            local = LocalUpdate(args=args, dataset=adjusted_datasets, idxs=read_dict_users[str(idx)]) # 对每个worker进行本地更新
            update, loss = local.gradient_train(net=copy.deepcopy(net_glob).to(args.device)) # 本地训练的weight和loss  ##第5行完成
            # torch.save(update, './cache/gradient_{}.pt'.format(idx))
            theta_i = OrderedDict()
            for key in w_glob.keys():
                theta_i[key] = update[key] + w_glob[key]
 
            gradient_locals.append(copy.deepcopy(update))
            loss_locals.append(copy.deepcopy(loss))
            local_models.append(copy.deepcopy(theta_i))  # Save local models for valuation

        local_data_volume = [len(read_dict_users[str(cid)]) for cid in idxs_users]
        total_data_volume = sum(local_data_volume)
        weights = [l_d_v / total_data_volume for l_d_v in local_data_volume]

        # CoAst参数裁剪
        pruned_updates = coast.prune_parameters(gradient_locals, w_glob_coast)
        # 裁剪参数聚合
        w_glob_valuation = coast.aggregate(pruned_updates, w_glob)

        # net_glob.load_state_dict(w_glob)

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
        

        # set = [i for i in range(args.num_users)]
        # ldv = [len(dict_users[cid]) for cid in set]
        # tdv = sum(ldv)
        # agg_weights = [i / tdv for i in ldv]
        # agg_parameter = [copy.deepcopy(gradient_locals[cid]) for cid in set]
        # # print(agg_parameter[0]['layer_input.weight'])
        # global_parameter = gradient_agg(agg_parameter, agg_weights, w_glob_shapley)
        # submodel_dict[list2str(set)].load_state_dict(global_parameter)
        # test_acc, test_loss = test_img(submodel_dict[list2str(set)], dataset_test, args, "test_dataset")
        # print(test_acc)



        """
        Shapley Calculation
        """

        

        # Calculate contributions
        contributions = coast.cross_round_valuation(local_models)
        print(contributions)

        shapley_dict = {}
        for i, score in enumerate(contributions):
            shapley_dict[i] = score

        for i, score in enumerate(contributions):
            total_contributions[i] += score
        
        with open('./save/CoAst.txt', 'a') as f:
            f.write('Training round:' + str(iter) + '\n')
            for i in shapley_dict:
                f.write('Client' + str(i) + '-> Shapley value: ' + str(float(shapley_dict[i])) + '\n')
        for i in range(args.num_users):
            total_shapley_dict[i] += shapley_dict[i]
        # print(shapley_dict)
        """
        Shapley Calculation
        """
    # print(total_shapley_dict)
    with open('./save/CoAst.txt', 'a') as f:
        f.write('\n\n')
        for key, value in total_shapley_dict.items():
            # 将键值对转换为字符串格式并写入文件
            f.write(f"{key}: {value}" + '\t')
    print(shapley_rank(total_shapley_dict))
    with open('./save/CoAst.txt', 'a') as f:
        f.write('\n')
        f.write('Shapley rank: ' + str(shapley_rank(total_shapley_dict)))


    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fedavg_{}_{}_{}_C{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, datetime.now().strftime("%H_%M_%S")))

    # testing  测试集上进行测试
    net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args, "train_dataset")
    acc_test, loss_test = test_img(net_glob, dataset_test, args, "test_dataset")

    # print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

    # with open('experiment.txt', 'a') as f:
    #     f.write('Exact_shapley:' + str(idxs_users) + str(acc_test)+'\n')

    end_time =  time.time()
    execution_time = end_time - start_time
    print("程序运行时间为：", execution_time, "秒")
    with open('./save/CoAst.txt', 'a') as f:
        f.write('Running_time: ' + str(execution_time) + 's'+'\n')

    with open('./shapley_result/running_time.txt', 'a') as f:
        f.write('CoAst: client number_' + str(args.num_users) + ' running_time: ' + str(execution_time) + 's'+'\n')
    # with open('./shapley_result/shapley_rank.txt', 'a') as f:
    #     f.write('Exact_shapley: client number_' + str(args.num_users) + ' shapley_rank: ' + str(shapley_rank(total_shapley_dict)) + '\n')