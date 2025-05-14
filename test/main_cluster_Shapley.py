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
from utils.utils import exp_details, worst_fraction, best_fraction
from utils.shapley import all_subsets, list2str, cal_shapley
from utils.cluster import load_clients, Kmeans_plusplus, k_means, cluster_agg, cluster_shapley_average_distribute, cluster_shapley_similarity_distribute


if __name__ == '__main__':
    args = args_parser()
    args.method = "cluster_shapley"
    n_cluster= int(args.num_users*0.6)
    exp_details(args)  # 打印超参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu') # 使用cpu还是gpu 赋值args.device

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
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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
    img_size = dataset_train[0][0].shape  # 图像的size

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
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

    with open('./save/Exact_Shapley.txt', 'a') as f:
        f.truncate(0)
    with open('./save/Margin_contribution_exactshapley.txt', 'a') as f:
        f.truncate(0)

    net_glob.train()
    w_glob = net_glob.state_dict()
    # 所有子集
    iterable = [i for i in range(n_cluster)]
    powerset = all_subsets(iterable)
    # print(powerset)
    # 初始化全部子模型
    submodel_dict = {}
    submodel_name_list = []
    
    # training
    loss_train_list = []

    # 本地数据量
    local_data_volume = [len(dict_users[cid]) for cid in range(args.num_users)]
    total_data_volume = sum(local_data_volume)
    # print(local_data_volume)
    # print(total_data_volume)
    total_shapley_dict = {}
    for i in range(args.num_users):
        total_shapley_dict[i] = 0
    
    for iter in range(args.epochs):
        for subset in powerset:
            # print(subset)
            # tuple_subset = tuple(subset)
            str_subset = list2str(subset)
            # print(str_subset)
            submodel_name_list.append(str_subset)
            submodel_dict[str_subset] = copy.deepcopy(net_glob)
            submodel_dict[str_subset].to(args.device)
            submodel_dict[str_subset].train()
        # print(submodel_name_list)
        # print(submodel_dict)

        loss_locals = []  # 对于每一个epoch，初始化worker的损失
        gradient_locals = []  # 存储客户端本地梯度

        net_glob.train()
        # m = max(int(args.frac * args.num_users), 1)
        # print(m)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print(idxs_users)
        idxs_users = [i for i in range(args.num_users)]
        
        for idx in idxs_users:  # 对于选取的m个worker
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) # 对每个worker进行本地更新
            update, loss = local.gradient_train(net=copy.deepcopy(net_glob).to(args.device)) # 本地训练的update和loss  ##第5行完成
            torch.save(update, './cache/gradient_{}.pt'.format(idx))
            
            local_data_volume = [len(dict_users[cid]) for cid in range(len(idxs_users))]
            total_data_volume = sum(local_data_volume)
            weights = [l_d_v / total_data_volume for l_d_v in local_data_volume]

            gradient_locals.append(copy.deepcopy(update))
            loss_locals.append(copy.deepcopy(loss))
        
        # print(gradient_locals)
        # clients = load_clients(gradient_locals)
        # print(clients)
        # clients_id = Kmeans_plusplus(gradients= gradient_locals,n_clients=args.num_users,device=args.device,epoch='N') # 使用kmeans++进行聚类
        
        # 聚类分组
        cluster_result = k_means(gradients= gradient_locals,n_clients=args.num_users,n_cluster= int(args.num_users*0.6),device=args.device)
        print(cluster_result)
        # 每个簇聚合一个代表梯度
        cluster_gradient = [[] for _ in range(n_cluster)]
        # print(cluster_gradient)
        for i in range(n_cluster):
            for j in cluster_result[i]:
                cluster_gradient[i].append(gradient_locals[j])
        # print(cluster_gradient)
        cluster_represent_gradient = []
        # print(cluster_represent_gradient)
        for i in range(n_cluster):
            cluster_represent_gradient.append(cluster_agg(cluster_gradient[i]))
        # print(cluster_represent_gradient[0])
        # print(len(cluster_represent_gradient))
        # print("###############")
        # print(gradient_locals[2])
        
        # 计算簇内数据量
        cluster_data_volume = []
        for i in cluster_result:
            # print(i)
            local_volume = 0
            for cid in i:
                # print(cid)
                local_volume += len(dict_users[cid])
                # print(local_volume)
            cluster_data_volume.append(local_volume)
        # print(cluster_data_volume)


        w_glob = gradient_agg(gradient_locals, weights, w_glob)

        # w_glob = weight_agg(w_locals, weights)
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train_list.append(loss_avg)

        # acc
        net_glob.eval()
        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        # print(acc_test, acc_train, loss_train, loss_test)

        # Shapley Calculation
        accuracy_dict = {}

        # print(len(w_locals))
        for set in powerset:
            if not set:
                test_acc, test_loss = test_img(submodel_dict[list2str(set)], dataset_test, args)
            else:    
                # print(set)
                # 聚合权重计算
                ldv = [cluster_data_volume[cid] for cid in set]
                # print(ldv)
                tdv = sum(ldv)
                # print(tdv)
                agg_weights = [i / tdv for i in ldv]
                # print(agg_weights)
                # 聚合参数
                agg_parameter = [copy.deepcopy(cluster_represent_gradient[cid]) for cid in set]
                # print(agg_parameter)
                global_parameter = gradient_agg(agg_parameter, agg_weights, w_glob)
                submodel_dict[list2str(set)].load_state_dict(global_parameter)
                test_acc, test_loss = test_img(submodel_dict[list2str(set)], dataset_test, args)
            accuracy_dict[list2str(set)] = test_acc
        
        accuracy_dict[submodel_name_list[-1]] = acc_test
        # print(accuracy_dict)
        cluster_shapley_dict = cal_shapley(accuracy_dict, n_cluster)
        print(cluster_shapley_dict)
        ### 将簇shapley贡献分配给簇内客户端
        # client_shapley_dict = cluster_shapley_average_distribute(args.num_users, cluster_shapley_dict, cluster_result)
        client_shapley_dict = cluster_shapley_similarity_distribute(args.num_users, cluster_shapley_dict, cluster_result, cluster_represent_gradient, gradient_locals)
        
        # 边际贡献写入txt文件
        with open('./save/Margin_contribution_clustershapley.txt', 'a') as f:
            f.write('Training round:' + str(iter) + '\n')
            for i in accuracy_dict:
                f.write('Client set ' + str(i) + '-> Margin contribution: ' + str(float(accuracy_dict[i])) + '\n')
        # SV写入txt文件
        with open('./save/Cluster_Shapley.txt', 'a') as f:
            f.write('Training round:' + str(iter) + '\n')
            for i in client_shapley_dict:
                f.write('Client' + str(i) + '-> Shapley value: ' + str(float(client_shapley_dict[i])) + '\n')
        for i in range(args.num_users):
            total_shapley_dict[i] += client_shapley_dict[i]
    print("Total shapley value of every clients:")
    print(total_shapley_dict)


    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fedavg_{}_{}_{}_C{}_iid{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid, datetime.now().strftime("%H_%M_%S")))

    # testing  测试集上进行测试
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)

    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))