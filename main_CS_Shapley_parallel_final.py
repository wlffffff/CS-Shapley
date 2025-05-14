# sourcery skip: identity-comprehension
import os
from scipy.special import comb
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
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
from utils.shapley import all_subsets, list2str, cal_shapley, shapley_rank


def load_gradient(path):
    """加载梯度"""
    gradient = torch.load(path, weights_only=False)
    return gradient

def gradient_to_vector(gradient_dict):
    """将梯度 OrderedDict 转换为展平的 numpy 向量"""
    return torch.cat([v.flatten() for v in gradient_dict.values()]).cpu().numpy()

def cal_shapley_i(utility, i, N):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    # print(i)
    # print(utility)
    sv = 0
    for key in utility:
        # print(key)
        if key == 'NULL':
            marginal_contribution = utility[str(i)] - utility['NULL']
            len_str = 0
            # print(str(i)+"- NULL")
            # print(len_str)
            sv += marginal_contribution /((comb(N-1,len_str))*N)
        else:
            if key != str(i):
                list_key = key.split('_')
                if str(i) in list_key:
                    continue
                if str(i) not in list_key:
                    list_key.append(str(i))
                int_list_key = [int(x) for x in list_key]  # 转为整数
                # print(int_list_key)
                list_key_sorted = sorted(int_list_key)
                # print(list_key_sorted)
                list_key_final = list(map(str, list_key_sorted))
                # print(list_key_final)
                # list_key.sort()
                new_str = '_'.join(list_key_final)
                # new_str = key + '_' + str(i)
                marginal_contribution = utility[new_str] - utility[key]
                # print(str(new_str+"-"+key))
                len_str = len(key.split('_'))
                # print(len_str)
                sv += marginal_contribution /((comb(N-1,len_str))*N)
    # print(shapley_dict)
    return sv

def process_user_shapley(i, args, selection_dict, dataset_info, gradient_locals, w_glob_state, util, local_data_volume, cos_sim_i):
    """处理单个用户的Shapley值计算(spawn兼容版本)"""
    # 在每个子进程中重新初始化必要的组件
    from models.Nets import MLP, CNNCifar, CharLSTM
    from models.test import test_img
    from models.qFed import gradient_agg
    from utils.shapley import all_subsets, list2str
    
    # print(net_glob_state)
    # 恢复模型状态
    if args.dataset == 'mnist':
        img_size = [1,28,28]
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.dataset == 'shakespeare':
        net_glob = CharLSTM().to(args.device)
    net_glob.load_state_dict(w_glob_state)
    
    w_glob = copy.deepcopy(w_glob_state)
    
    # 恢复数据集信息
    if dataset_info['name'] == 'mnist':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_test = datasets.MNIST('./data/mnist/', train=False, transform=trans)
    elif dataset_info['name'] == 'cifar':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=trans)
    elif dataset_info['name'] == 'shakespeare':
        dataset_test = ShakeSpeare(train=False)
    
    # indices = selection_dict[i]
    sum_cos_sim = sum(cos_sim_i)-1 
    # print(cos_sim_full[i, :])
    # print(sum_cos_sim)

    iterable = sorted(selection_dict[i])
    powerset = all_subsets(iterable)
    # print("计算客户端 {} 全集: {}".format(i, powerset))
    if args.overlook_ratio >= 1:
        powerset_final = powerset
    else:
        # 根据相似度，忽略部分子模型
        powerset_non_i = [subset for subset in powerset if i not in subset]
        # print(powerset_non_i)

        powerset_reverse = []
        for set in powerset_non_i:
            if set:
                # print(set)
                sum_cos = sum(cos_sim_i[j] for j in set)
                # print(sum_cos)
                if sum_cos > args.overlook_ratio * sum_cos_sim:
                    powerset_reverse.append(set)
                    # powerset_reverse.append(set.append(i))
        # print(powerset_reverse)

        powerset_reverse_final = []
        for set_r in powerset_reverse:
            # print(set_r)
            # 添加原始子集
            powerset_reverse_final.append(set_r)
            # print(powerset_reverse_final)
            # 生成扩展子集（插入0）
            extended_subset = sorted([i] + set_r.copy())  # 确保不修改原列表
            powerset_reverse_final.append(extended_subset)
        # print(powerset_reverse_final)

        powerset_final = [s for s in powerset if s not in powerset_reverse_final]   
        # print(powerset_final) 
    
    # 初始化子模型
    submodel_dict = {}
    for subset in powerset_final:
        str_subset = list2str(subset)
        submodel_dict[str_subset] = copy.deepcopy(net_glob)
        submodel_dict[str_subset].to(args.device)

    accuracy_dict = {}
    print(len(powerset_final))

    for subset in tqdm(powerset_final):
        # print(subset)
        set_key = list2str(subset)
        if util.get(set_key) is not None:
            accuracy_dict[set_key] = util[set_key]
        else:
            if not subset:
                test_acc, _ = test_img(submodel_dict[set_key], dataset_test, args, "test_dataset")
            else:
                ldv = [local_data_volume[cid] for cid in subset]
                tdv = sum(ldv)
                agg_weights = [i / tdv for i in ldv]

                agg_parameter = [copy.deepcopy(gradient_locals[cid]) for cid in subset]
                global_parameter = gradient_agg(agg_parameter, agg_weights, w_glob)
                submodel_dict[set_key].load_state_dict(global_parameter)
                test_acc, _ = test_img(submodel_dict[set_key], dataset_test, args, "test_dataset")
            accuracy_dict[set_key] = test_acc
            util[set_key] = test_acc
    # print(accuracy_dict)
    return i, cal_shapley_i(accuracy_dict, i, len(selection_dict[i]))


if __name__ == '__main__':
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn', force=True)

    args = args_parser()
    args.method = "cs_shapley_parallel"
    exp_details(args)  # 打印超参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu') # 使用cpu还是gpu 赋值args.device

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)  #训练集
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)  #测试集
        # y_train = np.array(dataset_train.targets)
        dataset_info = {'name': 'mnist'}
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
        dataset_info = {'name': 'cifar'}
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
        dataset_info = {'name': 'shakespeare'}
        # dict_users = dataset_train.get_client_dic()
        # args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    
    with open('./setup/dict_users.json', 'r') as f:
        original_read_dict_users = json.load(f)

    local_data_volume = {}
    for key in original_read_dict_users.keys():
        local_data_volume[int(key)] = len(original_read_dict_users[key])
    print(local_data_volume)

    # 创建共享字典
    manager = multiprocessing.Manager()
    local_data_volume_dict = manager.dict(local_data_volume)  # 转换为可共享的字典

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

    
    total_shapley_dict = {i: 0 for i in range(args.num_users)}

    start_time = time.time()

    gradient_dir = './gradient_cache'
    
    for iter in range(args.epochs):
        print("Evaluation round: {}".format(iter))

        w_glob = load_gradient(f'./gradient_cache/global_model/model_epoch_{iter}.pth')
        net_glob.load_state_dict(w_glob)
        
        idxs_users = [i for i in range(args.num_users)]

        gradient_list = []
        gradient_locals = []
        for cid in idxs_users:
            grad_path = os.path.join(gradient_dir, f'epoch_{iter}_client_{cid}.pt')
            gradient_locals.append(load_gradient(grad_path))
            gradient_list.append(load_gradient(grad_path))
        
        # 计算余弦相似度矩阵
        gradient_list = [gradient_to_vector(grad) for grad in gradient_list]
        cos_sim_sparse = lil_matrix((args.num_users, args.num_users))
        for i in range(args.num_users):
            for j in range(i, args.num_users):  # 只遍历上三角
                if i ==j:
                    cos_sim_sparse[i, j] = 1.0
                else:
                    cos_sim_sparse[i, j] = np.dot(gradient_list[i], gradient_list[j]) / (np.linalg.norm(gradient_list[i]) * np.linalg.norm(gradient_list[j])
        )
        cos_sim_full = np.triu(cos_sim_sparse.toarray()) + np.tril(cos_sim_sparse.toarray().T, k=-1)
        # print(cos_sim_full)


        # 选择客户端
        selection_dict = {}
        for i in range(args.num_users):
            similarity = cos_sim_full[i, :].flatten()
            # print(similarity)
            # similarity.sort()
            # print(similarity)
            index = heapq.nsmallest(int(args.num_users*args.selection_ratio)-1, range(len(similarity)), similarity.__getitem__)
            # print(index)
            selection_dict[i] = [i] + index
        # print(selection_dict)

        # 标准化相似度矩阵
        cos_sim_normalized = (cos_sim_full + 1) / 2
        # print(cos_sim_normalized)
        cos_sim_normalized_list = [None] * args.num_users
        for i in range(args.num_users):
            cos_sim_normalized_list[i] = cos_sim_normalized[i, :]

        # 使用多进程池并行计算
        manager = multiprocessing.Manager()
        util = manager.dict()
        shapley_dict = {i: 0 for i in range(args.num_users)}
        
        # 准备需要在进程间传递的状态
        w_glob_state = copy.deepcopy(w_glob)
        
        with ProcessPoolExecutor(max_workers=args.num_users) as executor:   # min(multiprocessing.cpu_count(), args.num_users)
        # with ProcessPoolExecutor(max_workers=min(multiprocessing.cpu_count(), args.num_users, 1)) as executor:
            futures = []
            for i in range(args.num_users):
                futures.append(executor.submit(
                    process_user_shapley,
                    i, args, selection_dict,
                    dataset_info, gradient_locals, w_glob_state, util, local_data_volume_dict, cos_sim_normalized_list[i]
                ))
            
            for future in as_completed(futures):
                try:
                    user_id, shapley_value = future.result()
                    shapley_dict[user_id] = shapley_value
                except Exception as e:
                    print(f"Error processing user {user_id}: {str(e)}")
                    shapley_dict[user_id] = 0  # 出错时赋默认值

        # SV写入txt文件
        with open('./save/CS_Shapley_Parallel.txt', 'a') as f:
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
    with open('./save/CS_Shapley_Parallel.txt', 'a') as f:
        f.write('\n\n')
        f.write('Selection ratio:' + str(args.selection_ratio) + 'Overlook ratio:' + str(args.overlook_ratio) + '\n')  
        for key, value in total_shapley_dict.items():
            # 将键值对转换为字符串格式并写入文件
            f.write(f"{key}: {value}" + '\t')
    print(shapley_rank(total_shapley_dict))
    with open('./save/CS_Shapley_Parallel.txt', 'a') as f:
        f.write('\n')
        f.write('Shapley rank: ' + str(shapley_rank(total_shapley_dict)))


    end_time =  time.time()
    execution_time = end_time - start_time
    print("程序运行时间为：", execution_time, "秒")
    with open('./save/CS_Shapley_Parallel.txt', 'a') as f:
        f.write('Running_time: ' + str(execution_time) + 's'+'\n')

    with open('./shapley_result/running_time.txt', 'a') as f:
        f.write('CS_Shapley_Parallel: client number_' + str(args.num_users) + ' running_time: ' + str(execution_time) + 's'+'\n')
    