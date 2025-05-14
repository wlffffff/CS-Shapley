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


if __name__ == '__main__':
    args = args_parser()
    args.method = "tmc_shapley"
    args.eps = 0.001 
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

    # with open('./save/TMC_Shapley.txt', 'a') as f:
    #     f.truncate(0)
    # with open('./save/Margin_contribution_tmcshapley.txt', 'a') as f:
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

    # 本地数据量
    # local_data_volume = [len(dict_users[cid]) for cid in range(args.num_users)]
    # total_data_volume = sum(local_data_volume)
    # print(local_data_volume)
    # print(total_data_volume)
    total_shapley_dict = {}
    for i in range(args.num_users):
        total_shapley_dict[i] = 0

    start_time = time.time()
    
    for iter in range(args.epochs):

        loss_locals = []  # 对于每一个epoch，初始化worker的损失
        gradient_locals = []  # 存储客户端本地权重

        ori_glob = copy.deepcopy(net_glob)
        model_weight = ori_glob.state_dict()
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

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        # loss_train_list.append(loss_avg)

        # # acc
        net_glob.eval()
        
        acc_test, loss_test = test_img(net_glob, dataset_test, args, "test_dataset")
        print("###############")
        # acc_train, loss_train = test_img(net_glob, dataset_train, args, "train_dataset")
        # # print(acc_test, acc_train, loss_train, loss_test)
        # print("Epoch {} Training accuracy: {:.2f}".format(iter, acc_train))
        print("Epoch {} Testing accuracy: {}".format(iter, acc_test))

        """
        Shapley Calculation
        """

        Contribution_records = []
        shapley_dict_old = {}
        for i in range(args.num_users):   # 初始化每个用户的SV为0
            shapley_dict_old[i] = 0

        # args.mc_iteration = int((2**args.num_users)/8)

        permutation = np.zeros((args.mc_iteration, args.num_users), int)
        selected_idx = [[]for _ in range(args.mc_iteration)]
        selected_idx_sorted = [[]for _ in range(args.mc_iteration)]
        marginal_contribs = np.zeros([args.num_users])

        for t in tqdm(range(args.mc_iteration)):
            permutation[t] = np.random.permutation(args.num_users)
            # print(permutation[t])
            marginal_contribution_t = {idx: 0 for idx in range(args.num_users)}
            
            selected_idx[t] = []

            non_value, _ = test_img(ori_glob, dataset_test, args, "test_dataset")
            # print(non_value)

            allset_value = acc_test
            # print(allset_value)
            v = np.zeros(args.num_users + 1)
            v[args.num_users] = non_value
            # print(v)

            for j in range(args.num_users):
                if abs(allset_value - v[j - 1]) < args.tolerance:
                    print("截断")
                    v[j] = v[j - 1]
                    # print(v[j])
                else:
                    # 一定要包含j
                    selected_idx[t].append(permutation[t][j])
                    selected_idx_sorted[t] = sorted(selected_idx[t])
                    # print(selected_idx[t])
                    ldv = [len(read_dict_users[str(cid)]) for cid in selected_idx[t]]
                    tdv = sum(ldv)
                    agg_weights = [i / tdv for i in ldv]
                    agg_parameter = [copy.deepcopy(gradient_locals[cid]) for cid in selected_idx[t]]
                    global_parameter = gradient_agg(agg_parameter, agg_weights, model_weight)
                    submodel_dict[list2str(selected_idx_sorted[t])].load_state_dict(global_parameter)
                    v[j], _ = test_img(submodel_dict[list2str(selected_idx_sorted[t])], dataset_test, args, "test_dataset")
                    # print(v[j])

                marginal_contribution_t[permutation[t][j]] = v[j] - v[j-1]
                marginal_contribs[permutation[t][j]] = t / (t + 1) * marginal_contribs[permutation[t][j]] + 1 / (t + 1) * (v[j] - v[j - 1])
            Contribution_records.append([marginal_contribution_t[i] for i in range(args.num_users)])

            # print("---------------------")
            # print(v)
            # print(marginal_contribs)

        shapley_value = (np.cumsum(Contribution_records, 0) /
                        np.reshape(np.arange(1, len(Contribution_records)+1), (-1, 1)))[-1:].tolist()[0]

        # store round t results
        shapley_dict = {key: sv for key, sv in enumerate(shapley_value)}
        # print(shapley_dict)

        for i in range(args.num_users):
            shapley_dict_old[i] += marginal_contribs[i]


        # SV写入txt文件
        with open('./save/TMC_Shapley.txt', 'a') as f:
            f.write('Training round:' + str(iter) + '\n')
            for i in shapley_dict:
                f.write('Client' + str(i) + '-> Shapley value: ' + str(float(shapley_dict[i])) + '\n')
        for i in range(args.num_users):
            total_shapley_dict[i] += shapley_dict[i]
    
    print(total_shapley_dict)
    with open('./save/TMC_Shapley.txt', 'a') as f:
        f.write('\n\n')
        for key, value in total_shapley_dict.items():
            # 将键值对转换为字符串格式并写入文件
            f.write(f"{key}: {value}" + '\t')
    print(shapley_rank(total_shapley_dict))
    with open('./save/TMC_Shapley.txt', 'a') as f:
        f.write('\n')
        f.write(str(args.mc_iteration) + 'Shapley rank: ' + str(shapley_rank(total_shapley_dict)))
    """
    Shapley Calculation
    """


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

    with open('./shapley_result/running_time.txt', 'a') as f:
        f.write('TMC_shapley: client number_' + str(args.num_users) + ' iteration_' + str(args.mc_iteration) + ' running_time: ' + str(execution_time) + 's'+'\n')
    # with open('./shapley_result/shapley_rank.txt', 'a') as f:
    #     f.write('TMC_shapley: client number_' + str(args.num_users) + ' iteration_' + str(args.mc_iteration) + ' shapley_rank: ' + str(shapley_rank(total_shapley_dict)) + '\n')