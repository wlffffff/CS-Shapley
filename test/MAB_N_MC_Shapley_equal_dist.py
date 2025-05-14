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
import math
import heapq
import json
from scipy.stats import norm

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
from utils.shapley import all_subsets, list2str, cal_shapley, cal_mab_stratified_shapley_0, cal_mab_stratified_shapley_t, shapley_rank, cal_mab_stratified_shapley_t_N


def normal_distribution(data_list):
    data_array = np.array(data_list)
 
    # 计算均值和标准差
    mean_value = data_array.mean()
    std_dev = data_array.std()
    
    # # 创建正态分布对象
    # normal_dist = norm(loc=mean_value, scale=std_dev)
    
    # # 打印出正态分布的参数
    # print(f"均值: {mean_value}, 标准差: {std_dev}")

    # plt.hist(data_list, bins=50, alpha=0.6, color='g')
    # xmin, xmax = plt.xlim()
    # # print(xmin, xmax)
    # x = np.linspace(xmin, xmax, 100)
    # p = norm.pdf(x, mean_value, std_dev)
    # plt.plot(x, p, 'k', linewidth=2)
    # plt.show()

    norm_cdf = []
    sum_norm_cdf = 0

    for data in data_list:
        cdf_value = norm.cdf(data, mean_value, std_dev)
        norm_cdf.append(cdf_value)
        sum_norm_cdf += cdf_value
        # print(cdf_value)

    # print(norm_cdf)
    # print(sum_norm_cdf)

    norm_weight_list = [element / sum_norm_cdf for element in norm_cdf]
    # print(norm_weight_list)
    return (norm_weight_list)

def equal_distribution(data_list):
    # data_array = np.array(data_list)
 
    norm_weight_list = [1 / len(data_list) for _ in range(len(data_list))]
    # print(norm_weight_list)
    return (norm_weight_list)

def normalization_distribution(data_list):
    # data_array = np.array(data_list)
    sum_list = sum(data_list)
 
    norm_weight_list = [i / sum_list for i in data_list]
    # print(norm_weight_list)
    return (norm_weight_list)


def mean_func(data_list):
    return 0 if data_list == [] else np.mean(data_list)


if __name__ == '__main__':
    args = args_parser()
    args.method = "mab_mc_shapley_nonnd"
    exp_details(args)  # 打印超参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu') # 使用cpu还是gpu 赋值args.device

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)  #训练集
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)  #测试集
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

    # with open('./save/MabStratifiedMC_Shapley.txt', 'a') as f:
    #     f.truncate(0)
    # with open('./save/Margin_contribution_mabstratifiedmcshapley.txt', 'a') as f:
    #     f.truncate(0)

    net_glob.train()
    w_glob = net_glob.state_dict()
    # 所有子集
    iterable = [i for i in range(args.num_users)]
    powerset = all_subsets(iterable)    # e.g. [[],[0],[1],[0.1]]
    # print(powerset)
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
        submodel_dict[str_subset].train()
    # print(submodel_name_list)
    # print(submodel_dict)

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

    SVR = np.zeros(args.num_users)  # 计算排列累计重要性
    T = np.zeros(args.num_users)  # 计算排列采样次数
    B = np.zeros(args.num_users)  # 计算每个客户端的每个排列的B值
    # print(T)
    # print(SVR)
    # print(B)
    
    for iter in range(args.epochs):
        """
        Shapley Calculation
        """
        
        w_glob_shapley = copy.deepcopy(w_glob)
        """
        Shapley Calculation
        """

        loss_locals = []  # 对于每一个epoch，初始化worker的损失
        gradient_locals = []  # 存储客户端本地权重

        net_glob.train()
        # m = max(int(args.frac * args.num_users), 1)
        # print(m)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print(idxs_users)
        idxs_users = [i for i in range(args.num_users)]
        
        for idx in idxs_users:  # 对于选取的m个worker
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=read_dict_users[str(idx)]) # 对每个worker进行本地更新
            update, loss = local.gradient_train(net=copy.deepcopy(net_glob).to(args.device)) # 本地训练的weight和loss  ##第5行完成
            # torch.save(update, './cache/gradient_{}.pt'.format(idx))
 
            gradient_locals.append(copy.deepcopy(update))
            loss_locals.append(copy.deepcopy(loss))

        local_data_volume = [len(read_dict_users[str(cid)]) for cid in idxs_users]
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
        # print("###############")
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
        accuracy_dict = {}
        
        if iter == 0:             # 第一轮需要采样所有排列，因此需要聚合所有可能的子模型，并获得其准确率
            # print(len(w_locals))
            for my_set in tqdm(powerset):
                # print(my_set)
                if not my_set:
                    ### 空集合就直接测试原始模型的性能
                    test_acc, _ = test_img(submodel_dict[list2str(my_set)], dataset_test, args, "test_dataset")
                    # print("not set")
                else:
                    # print(my_set)
                    # 聚合权重计算
                    ldv = [len(read_dict_users[str(cid)]) for cid in my_set]
                    # print(ldv)
                    tdv = sum(ldv)
                    # print(tdv)
                    agg_weights = [i / tdv for i in ldv]
                    # print(agg_weights)
                    # 聚合参数
                    agg_parameter = [copy.deepcopy(gradient_locals[cid]) for cid in my_set]
                    # print(agg_parameter)
                    global_parameter = gradient_agg(agg_parameter, agg_weights, w_glob_shapley)
                    submodel_dict[list2str(my_set)].load_state_dict(global_parameter)
                    test_acc, _ = test_img(submodel_dict[list2str(my_set)], dataset_test, args, "test_dataset")
                accuracy_dict[list2str(my_set)] = test_acc   
            # print(accuracy_dict)
            shapley_dict, client_stra_importance = cal_mab_stratified_shapley_0(accuracy_dict, args.num_users)

            ratio = copy.deepcopy(client_stra_importance)
            array = np.array(ratio)
            column_sums = np.sum(array, axis=0).tolist()  # 计算每一列的和:分层的重要性
            # print(column_sums)

            all_sums = sum(column_sums)
            sum_ratio = [i / all_sums for i in column_sums]

            # for i in range(args.num_users):
            #     sum_column = sum(client_stra_importance[i])
            #     # print(sum_column)
            #     for j in range(args.num_users):
            #         ratio[i][j] = client_stra_importance[i][j]/sum_column
            # # print(ratio)
            # sum_ratio = np.cumsum(ratio, 0)[-1:].tolist()[0]
            # print(sum_ratio)

            # 更新SVR, T, B
            SVR += sum_ratio
            
            for j in range(args.num_users):
                T[j] += 1
                B[j] = -math.sqrt(T[j]/args.para_H)+SVR[j]
            # print(SVR)
            # print(T)
            # print(B)

            # 选择Top-k个B作为下一轮采样排列的指导
            
            index = sorted(heapq.nlargest(int(args.num_users*args.para_K), range(len(B)), B.__getitem__))
            print(index)
            SVR_index = [SVR[i] for i in index]
            print(SVR_index)
            sample_size = equal_distribution(SVR_index)     # 用于确定每个分层的采样数量
            print(sample_size)
        else:
            # print(index)
            sample_size_new = [int(args.iteration * i) for i in sample_size]
            print(sample_size_new)
            sample_size_final = [0 for _ in range(args.num_users)]
            for i, j in zip(index, range(len(sample_size_new))):
                sample_size_final[i] = sample_size_new[j]
            print(sample_size_final)

            importance_k = []

            Contribution_records = [[[] for _ in range(args.num_users)] for _ in range(args.num_users)]
            # print(Contribution_records)

            util = {}    # 相当于缓存池，保存子模型的准确率

            iteration_list = []    # 选择的每个分层的采样数量
            for i in index:
                iteration_list.append(sample_size_final[i])
            # print(iteration_list)

            shapley_dict = {i: 0 for i in range(args.num_users)}

            # sorted_selected_strata = sorted(index)
            # position = []
            # for i in sorted_selected_strata:
            #     position.extend((i, i+1))

            marginal_contribs = np.zeros((max(iteration_list), args.num_users))

            iteration_num = {item: 0 for item in index}
            for t in tqdm(range(max(iteration_list))):
                permutation = np.random.permutation(args.num_users)
                # print(permutation)
                for j, break_point in zip(index, iteration_list):
                    # print(j)
                    # print(break_point)
                    # print(iteration_num[j])
                    if iteration_num[j] < break_point:
                        iteration_num[j] += 1
                        C = permutation[:j]
                        C = tuple(np.sort(C, kind='mergesort'))
                        # print(C)
                        if util.get(C) != None:
                            V_C = util[C]
                            # print(V_C)
                        else:
                            if C == ():
                                V_C, _ = test_img(submodel_dict[list2str(C)], dataset_test, args, "test_dataset")
                            else:
                                ldv = [len(read_dict_users[str(cid)]) for cid in C]
                                tdv = sum(ldv)
                                agg_weights = [i / tdv for i in ldv]
                                agg_parameter = [copy.deepcopy(gradient_locals[cid]) for cid in C]
                                global_parameter = gradient_agg(agg_parameter, agg_weights, w_glob_shapley)
                                submodel_dict[list2str(C)].load_state_dict(global_parameter)
                                V_C, _ = test_img(submodel_dict[list2str(C)], dataset_test, args, "test_dataset")
                            # print(V_C)
                            util[C] = V_C
                        C_add_1 = permutation[:j+1]
                        C_add_1 = tuple(np.sort(C_add_1, kind='mergesort'))
                        # print(C_add_1)
                        if util.get(C_add_1) != None:
                            V_C_add_1 = util[C_add_1]
                            # print(V_C_add_1)
                        else:
                            ldv = [len(read_dict_users[str(cid)]) for cid in C_add_1]
                            tdv = sum(ldv)
                            agg_weights = [i / tdv for i in ldv]
                            agg_parameter = [copy.deepcopy(gradient_locals[cid]) for cid in C_add_1]
                            global_parameter = gradient_agg(agg_parameter, agg_weights, w_glob_shapley)
                            submodel_dict[list2str(C_add_1)].load_state_dict(global_parameter)
                            V_C_add_1, _ = test_img(submodel_dict[list2str(C_add_1)], dataset_test, args, "test_dataset")
                            # print(V_C_add_1)
                            util[C_add_1] = V_C_add_1
                        # update SV
                        marginal_contribs[permutation[j]] = V_C_add_1 - V_C
                        # print(f"client{permutation[j]}: {list2str(C_add_1)}-{list2str(C)}")
                        Contribution_records[j][permutation[j]].append(V_C_add_1 - V_C)
                        # print(Contribution_records)        

            for client in range(args.num_users):
                phi_client = []
                for j in index:
                    shapley_dict[int(client)] += 1/args.num_users * (mean_func(Contribution_records[j][client]))
                    # phi_client.append(1/args.num_users * mean_func(Contribution_records[j][client]))
                    phi_client.append(mean_func(Contribution_records[j][client]))
                importance_k.append(phi_client)
            
            # print(shapley_dict)
            # print(importance_k)

            client_stra_importance = importance_k


            ratio = copy.deepcopy(client_stra_importance)
            array = np.array(ratio)
            column_sums = np.sum(array, axis=0).tolist()  # 计算每一列的和:分层的重要性
            # print(column_sums)

            all_sums = sum(column_sums)
            # print(all_sums)
            sum_ratio_t = [i / all_sums for i in column_sums]

            # print(sum_ratio_t)

            new_sum_ratio = [0 for _ in range(args.num_users)]
            for i, j in zip(sum_ratio_t, index):
                new_sum_ratio[j] = i

            # 更新SVR, T, B
            SVR += new_sum_ratio
            for j in index:
                T[j] += 1
                B[j] = -math.sqrt(T[j]/args.para_H)+SVR[j]
            # print(SVR)
            # print(T)
            # print(B)

            index = sorted(heapq.nlargest(int(args.num_users*args.para_K), range(len(B)), B.__getitem__))
            # print(index)
            SVR_index = [SVR[i] for i in index]
            # print(SVR_index)
            sample_size = equal_distribution(SVR_index)     # 用于确定每个分层的采样数量
            # print(sample_size)

        """
        保存shapley信息到txt文件(边际贡献、SV值)
        """
        # # 边际贡献写入txt文件
        # with open('./save/Margin_contribution_mabstratifiedmcshapley.txt', 'a') as f:
        #     f.write('Training round:' + str(iter) + '\n')
        #     for i in accuracy_dict:
        #         f.write('Client set ' + str(i) + '-> Margin contribution: ' + str(float(accuracy_dict[i])) + '\n')
        # SV写入txt文件
        with open('./save/Mab_N_MC_Shapley_equal_dist.txt', 'a') as f:
            f.write('Training round:' + str(iter) + '\n')
            for i in shapley_dict:
                f.write('Client' + str(i) + '-> Shapley value: ' + str(float(shapley_dict[i])) + '\n')
        for i in range(args.num_users):
            total_shapley_dict[i] += shapley_dict[i]
        # print(shapley_dict)
        
    print(total_shapley_dict)
    with open('./save/Mab_N_MC_Shapley_equal_dist.txt', 'a') as f:
        f.write('\n\n')
        f.write("Parameter H:" + str(args.para_H) + " Parameter K:" + str(args.para_K) + " Parameter iteration:" + str(args.iteration) + '\n')
        for key, value in total_shapley_dict.items():
            # 将键值对转换为字符串格式并写入文件
            f.write(f"{key}: {value}" + '\t')
        f.write('\n')
        f.write('Shapley rank: ' + str(shapley_rank(total_shapley_dict)))
        f.write('\n')
    print(shapley_rank(total_shapley_dict))

    """
    shapley计算完成
    """

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

    end_time =  time.time()
    execution_time = end_time - start_time
    print("程序运行时间为：", execution_time, "秒")
    with open('./save/Mab_N_MC_Shapley_equal_dist.txt', 'a') as f:
        f.write('Running_time: ' + str(execution_time) + 's'+'\n')

    with open('./shapley_result/running_time.txt', 'a') as f:
        f.write('Mab_N_mc_shapley_equal_dist: client number_' + str(args.num_users) + ' selected strat_' + str(int(args.num_users* args.para_K)) + ' para_H_' + str(args.para_H) + " Para_iteration_" + str(args.iteration) +
                 ' running_time: ' + str(execution_time) + 's'+'\n')    # with open('./shapley_result/shapley_rank.txt', 'a') as f:
    #     f.write('Mab_stratified_mc_shapley: client number_' + str(args.num_users) + ' selected strat_' + str(int(args.num_users* args.para_K)) + ' shapley_rank: ' + str(shapley_rank(total_shapley_dict)) + '\n')