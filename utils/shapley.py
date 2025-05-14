import itertools
from scipy.special import comb
import numpy as np
import math
import heapq
import random
import copy
from collections import OrderedDict
 
def all_subsets(s):
    return [list(subset) for i in range(len(s) + 1) for subset in itertools.combinations(s, i)]

def list2str(list):  # [0,1,2]
    if not list:
        result = 'NULL'
    else:    
        result = '_'.join(map(str, list))
    return result

def str_option2list(str_option):
    result = []
    for i in range(len(str_option)):
        str = str_option[i].split('-')
        # print(str)
        for j in str:
            if j == 'NULL':
                result.append([])
            elif '_' in j:
                a = j.split('_')
                # print(a)
                int_list =  [int(item) for item in a]
                result.append(int_list)
            else:
                result.append([int(j)])
    # print(result)
    return result


def shapley(utility, N):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    shapley_dict = {}
    for i in range(1,N+1):
        shapley_dict[i] = 0
    for key in utility:
        print(key)
        if key != ():
            for contributor in key:
                print('contributor:', contributor, key) # print check
                marginal_contribution = utility[key] - utility[tuple(i for i in key if i!=contributor)]
                print('marginal:', marginal_contribution) # print check
                shapley_dict[contributor] += marginal_contribution /((comb(N-1,len(key)-1))*N)
    return shapley_dict

def cal_shapley(utility, N):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    shapley_dict = {}
    for i in range(N):   # 初始化每个用户的SV为0
        shapley_dict[i] = 0
    for key in utility:
        # print(key)
        if key != 'NULL':
            list_key = key.split('_')
            # print(list_key)
            # print(comb(N-1,len(list_key)-1))
            for contributor in list_key:  # contributor: NULL,0,1,2,0_1,0_2,1_2,0_1_2 
                # print('contributor to coalition', contributor, list_key) # print check
                if len(list_key) == 1:
                    marginal_contribution = utility[key] - utility['NULL']
                elif len(list_key) == 2:
                    for i in list_key:
                        if i!=contributor:
                            j = i 
                    marginal_contribution = utility[key] - utility[j]
                else:
                    lst = []
                    for i in list_key:
                        if i!=contributor:
                            lst.append(i)
                    str_key = list2str(lst)
                    marginal_contribution = utility[key] - utility[str_key]
                # print('marginal:', marginal_contribution) # print check
                shapley_dict[int(contributor)] += marginal_contribution /((comb(N-1,len(list_key)-1))*N)
                # print(int(contributor), marginal_contribution /((comb(N-1,len(list_key)-1))*N), len(list_key))
                # if(len(list_key)>2):
                #     print(str_key)
    # print(shapley_dict)
    return shapley_dict

def cal_stratified_shapley(utility, N):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    shapley_dict = {}
    for i in range(N):   # 初始化每个用户的SV为0
        shapley_dict[i] = 0
    for client in range(N):
        # print(client)
        iterable = [i for i in range(N)]
        # print(iterable)
        new_iterable = list(filter(lambda x: x != client, iterable))
        # print(new_iterable)
        powerset = all_subsets(new_iterable)
        # print(powerset)
        stratum = []
        for j in range(N):
            lst = []
            for element in powerset:
                if len(element) == j:
                    lst.append(element)
            stratum.append(lst)
        # print(stratum)
        for j in range(N):
            if j == 0:
                marginal_contribution = utility[str(client)] - utility['NULL']
                shapley_dict[int(client)] += marginal_contribution / comb(N-1,j)
            else:
                for coalition in stratum[j]:
                    # print(coalition)
                    new_coalition = copy.deepcopy(coalition)
                    new_coalition.append(client)
                    # print(sorted(new_coalition))
                    marginal_contribution = utility[list2str(sorted(new_coalition))] - utility[list2str(coalition)]
                    shapley_dict[int(client)] += marginal_contribution / comb(N-1,j)
    for key, value in shapley_dict.items():
        shapley_dict[key] = value/N
    # print(shapley_dict)
    return shapley_dict


def cal_delta_shapley(utility, N, strata, powerset):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    shapley_dict = {}
    for i in range(N):   # 初始化每个用户的SV为0
        shapley_dict[i] = 0
    
    for client in range(N):
        stratum = []
        for j in range(N):
            lst = []
            if j in strata:
                for element in powerset:
                    if len(element) == j and client not in element:
                        lst.append(element)
            stratum.append(lst)
        # print(stratum)

        for j in strata[:-1]:
            if j == 0:
                marginal_contribution = utility[str(client)] - utility['NULL']
                shapley_dict[int(client)] += marginal_contribution / (len(strata)-1)
            else:
                # print(stratum[j])
                for coalition in stratum[j]:
                    # print(coalition)
                    # print(utility[list2str(coalition)])
                    new_coalition = copy.deepcopy(coalition)
                    new_coalition.append(client)
                    # print(sorted(new_coalition))
                    # print(utility[list2str(sorted(new_coalition))])
                    marginal_contribution = utility[list2str(sorted(new_coalition))] - utility[list2str(coalition)]
                    # print(marginal_contribution)
                    # print(marginal_contribution / (len(strata)-1))
                    shapley_dict[int(client)] += marginal_contribution / (len(strata)-1)
    # for key, value in shapley_dict.items():
    #     shapley_dict[key] = value
    # print(shapley_dict)
    return shapley_dict


def cal_mab_stratified_shapley_0(utility, N):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    importance_k = []
    shapley_dict = {}
    for i in range(N):   # 初始化每个用户的SV为0
        shapley_dict[i] = 0
    for client in range(N):
        phi_client = []
        # print(client)
        iterable = [i for i in range(N)]
        # print(iterable)
        new_iterable = list(filter(lambda x: x != client, iterable))
        # print(new_iterable)
        powerset = all_subsets(new_iterable)
        # print(powerset)
        stratum = []
        for j in range(N):
            lst = [element for element in powerset if len(element) == j]
            stratum.append(lst)
        # print(stratum)
        for j in range(N):
            if j == 0:
                marginal_contribution = utility[str(client)] - utility['NULL']
                phi_client_0 = marginal_contribution / (comb(N-1,j))
                shapley_dict[int(client)] += marginal_contribution / (N*comb(N-1,j))
                phi_client.append(phi_client_0)
            else:
                phi_client_k = 0
                for coalition in stratum[j]:
                    # print(coalition)
                    new_coalition = copy.deepcopy(coalition)
                    new_coalition.append(client)
                    # print(sorted(new_coalition))
                    marginal_contribution = utility[list2str(sorted(new_coalition))] - utility[list2str(coalition)]
                    phi_client_k += marginal_contribution / (comb(N-1,j))
                    shapley_dict[int(client)] += marginal_contribution / (N*comb(N-1,j))
                phi_client.append(phi_client_k)
        importance_k.append(phi_client)
    # print(shapley_dict)
    # print(importance_k)
    return shapley_dict, importance_k

def find_index(lst, condition):
    for i, x in enumerate(lst):
        if len(x[0])==condition:
            return i 

def cal_mab_stratified_shapley_t(utility, N, selected_stra):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    importance_k = []
    shapley_dict = {}
    for i in range(N):   # 初始化每个用户的SV为0
        shapley_dict[i] = 0
    for client in range(N):
        phi_client = []
        # print(client)
        iterable = [i for i in range(N)]
        # print(iterable)
        new_iterable = list(filter(lambda x: x != client, iterable))
        # print(new_iterable)
        powerset = all_subsets(new_iterable)
        # print(powerset)
        stratum = []
        for j in selected_stra:
            lst = [element for element in powerset if len(element) == j]
            stratum.append(lst)
        # print(stratum)
        for j in selected_stra:
            if j == 0:
                marginal_contribution = utility[str(client)] - utility['NULL']
                # print(str(client))
                # print('NULL')
                phi_client_0 = marginal_contribution / (len(selected_stra)*comb(N-1,j))
                shapley_dict[int(client)] += marginal_contribution / (len(selected_stra)*comb(N-1,j))
                phi_client.append(phi_client_0)
            else:
                phi_client_k = 0
                m = find_index(stratum, j)
                for coalition in stratum[m]:
                    # print(coalition)
                    new_coalition = copy.deepcopy(coalition)
                    new_coalition.append(client)
                    # print(sorted(new_coalition))
                    marginal_contribution = utility[list2str(sorted(new_coalition))] - utility[list2str(coalition)]
                    # print(list2str(sorted(new_coalition)))
                    # print(list2str(coalition))
                    phi_client_k += marginal_contribution / (len(selected_stra)*comb(N-1,j))
                    shapley_dict[int(client)] += marginal_contribution / (len(selected_stra)*comb(N-1,j))
                phi_client.append(phi_client_k)
        importance_k.append(phi_client)
    # print(shapley_dict)
    # print(importance_k)
    return shapley_dict, importance_k

def cal_mab_stratified_shapley_t_N(utility, N, selected_stra):
    """
    :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
    and the values are the accuracies from training on a combination of these trainsets
    :param N: total number of data contributors
    :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
    """
    importance_k = []
    shapley_dict = {}
    for i in range(N):   # 初始化每个用户的SV为0
        shapley_dict[i] = 0
    for client in range(N):
        phi_client = []
        # print(client)
        iterable = [i for i in range(N)]
        # print(iterable)
        new_iterable = list(filter(lambda x: x != client, iterable))
        # print(new_iterable)
        powerset = all_subsets(new_iterable)
        # print(powerset)
        stratum = []
        for j in selected_stra:
            lst = [element for element in powerset if len(element) == j]
            stratum.append(lst)
        # print(stratum)
        for j in selected_stra:
            if j == 0:
                marginal_contribution = utility[str(client)] - utility['NULL']
                # print(str(client))
                # print('NULL')
                phi_client_0 = marginal_contribution / (N*comb(N-1,j))
                shapley_dict[int(client)] += marginal_contribution / (N*comb(N-1,j))
                phi_client.append(phi_client_0)
            else:
                phi_client_k = 0
                m = find_index(stratum, j)
                for coalition in stratum[m]:
                    # print(coalition)
                    new_coalition = copy.deepcopy(coalition)
                    new_coalition.append(client)
                    # print(sorted(new_coalition))
                    marginal_contribution = utility[list2str(sorted(new_coalition))] - utility[list2str(coalition)]
                    # print(list2str(sorted(new_coalition)))
                    # print(list2str(coalition))
                    phi_client_k += marginal_contribution / (N*comb(N-1,j))
                    shapley_dict[int(client)] += marginal_contribution / (N*comb(N-1,j))
                phi_client.append(phi_client_k)
        importance_k.append(phi_client)
    # print(shapley_dict)
    # print(importance_k)
    return shapley_dict, importance_k

def p_t_cal(list, N):
    sorted_list = sorted(list)
    # print(sorted_list)
    sorted_list = [j +1 for j in sorted_list]
    # print(sorted_list)
    split_list = []
    result = []
    p_t = []
    ori = 0
    for i in range(N-1):
        ori += comb(N-1,i)
        split_list.append(int(ori))
    # print(split_list)
    for i in split_list:
        for index, value in enumerate(sorted_list):
            if i < value:
                result.append(sorted_list[:index])
                sorted_list = sorted_list[index:]
                break
    result.append(sorted_list)
    # print(result)   
    sum = 0
    for i in range(len(result)):
        p_t.append(len(result[i]))
    # print(p_t)
    p_t = [x for x in p_t if x != 0]
    # print(p_t)
    final_p_t = []
    for i in p_t:
        if i == 1:
            final_p_t.append(1/(i*len(p_t)))
        else:
            for _ in range(i):
                final_p_t.append(1/(i*len(p_t)))
    # print(final_p_t)
    return final_p_t


# def mab_based_cal_shapley(utility, N, round, para_H, para_K):
#     """
#     :param utility: a dictionary with keys being tuples. (1,2,3) means that the trainset 1,2 and 3 are used,
#     and the values are the accuracies from training on a combination of these trainsets
#     :param N: total number of data contributors
#     :returns the dictionary with the shapley values of the data, eg: {1: 0.2, 2: 0.4, 3: 0.4}
#     """
#     shapley_dict = {}
#     permutation_value = [[]for _ in range(N)]
#     permutation_key = [i for i in range(N)]
#     permutation_dict = {}
#     pertumation_num = [[] for _ in range(N)]
#     # print(pertumation_num)
#     MC_t = np.zeros((N, 2**(N-1)))  # 计算第t轮每个排列的重要性
#     SVR = np.zeros((N, 2**(N-1)))  # 计算排列累计重要性
#     T = np.zeros((N, 2**(N-1)))  # 计算排列采样次数
#     B = np.zeros((N, 2**(N-1)))  # 计算每个客户端的每个排列的B值

#     for i in range(N):   # 初始化每个用户的SV为0
#         shapley_dict[i] = 0
    
#     if round == 0:
#         for key in utility:
#             print(key)
#             if key != 'NULL':
#                 list_key = key.split('_')
#                 print(list_key)
#                 # print(comb(N-1,len(list_key)-1))
#                 for contributor in list_key:  # contributor: NULL,0,1,2,0_1,0_2,1_2,0_1_2 
#                     print('contributor to coalition', contributor, list_key) # print check
#                     if len(list_key) == 1:
#                         marginal_contribution = utility[key] - utility['NULL']
#                         pertumation_num[int(contributor)].append(str(key)+'-'+'NULL')
#                     elif len(list_key) == 2:
#                         for i in list_key:
#                             if i!=contributor:
#                                 j = i 
#                         marginal_contribution = utility[key] - utility[j]
#                         pertumation_num[int(contributor)].append(str(key)+'-'+str(j))
#                     else:
#                         lst = []
#                         for i in list_key:
#                             if i!=contributor:
#                                 lst.append(i)
#                             str_key = list2str(lst)
#                         marginal_contribution = utility[key] - utility[str_key]
#                         pertumation_num[int(contributor)].append(str(key)+'-'+str(str_key))
#                     # print('marginal:', marginal_contribution) # print check
#                     shapley_dict[int(contributor)] += marginal_contribution /((comb(N-1,len(list_key)-1))*N)
#                     permutation_value[int(contributor)].append(marginal_contribution /((comb(N-1,len(list_key)-1))*N))
#         # print(pertumation_num)
        
#         for i in range(len(permutation_key)):
#             permutation_dict[permutation_key[i]] = permutation_value[i]  # {0: [25.0, 12.5, ...], 1: [30.0, 14.166666666666666, ...], ...,3: [20.0, 12.5, 15.0, ...]}    
#         # print(permutation_dict)  # {0: [25.0, 12.5, ...], 1: [30.0, 14.166666666666666, ...], ...,3: [20.0, 12.5, 15.0, ...]}
#         for i in range(N):
#             for j in range(len(permutation_dict[0])):
#                 MC_t[i][j]=permutation_dict[i][j] / sum(permutation_dict[i])
#                 T[i][j]=T[i][j]+1
#                 SVR[i][j]+=MC_t[i][j]
#                 B[i][j]=-math.sqrt(T[i][j]/para_H)+SVR[i][j]
#         # print(MC_t)
#         # print(T)
#         # print(SVR)
#         # print(B)

#         # 选择Top-k个B作为下一轮采样排列的指导
#         B_list = []
#         selected_per_list = []
#         p_t_list = []
#         for i in range(N):
#             B_list.append(B[i])
#             a = sorted(heapq.nlargest(para_K, range(len(B[i])), B[i].__getitem__))
#             selected_per_list.append(a)
#             # print(a)
#             p_t = p_t_cal(a, 3)
#             p_t_list.append(p_t)

#     else:
#         # B = np.random.random((N, 2**(N-1)))
#         # print(B)
#         for i in range(N):
#             mc_list = []
#             for j in range(para_K):
#                 # print(pertumation_num[i][a[j]])
#                 permutation = pertumation_num[i][a[j]].split('-')
#                 # print(permutation)
#                 marginal_contribution = utility[permutation[0]] - utility[permutation[1]]
#                 # print(marginal_contribution)
#                 mc_list.append(marginal_contribution)
#             # print(mc_list)
#             weighted_mc = [x * y for x, y in zip(p_t, mc_list)]
#             # print(weighted_mc)
#             shapley_dict[i] = sum(weighted_mc)
#             for k in range(para_K):
#                 MC_t[i][a[k]]=weighted_mc[k] / sum(weighted_mc)
#                 # print(MC_t[i][a[k]])
#                 T[i][a[k]]=T[i][a[k]]+1
#                 # print(T[i][a[k]])
#                 SVR[i][a[k]]+=MC_t[i][a[k]]
#                 # print(SVR[i][a[k]])
#                 B[i][a[k]]=-math.sqrt(T[i][a[k]]/para_H)+SVR[i][a[k]]
#                 # print(B[i][a[k]])
#         # print(MC_t)
#         # print(T)
#         # print(SVR)
#         # print(B)    



#     # print(permutation_value)
#     # print(len(permutation_value))
#     # print(shapley_dict)
#     return shapley_dict, pertumation_num, B_list, selected_per_list, p_t_list 

def mab_based_cal_shapley_0(utility, N):
    shapley_dict = {}
    permutation_value = [[]for _ in range(N)]
    permutation_key = [i for i in range(N)]
    permutation_dict = {}
    pertumation_num = [[] for _ in range(N)]

    for i in range(N):   # 初始化每个用户的SV为0
        shapley_dict[i] = 0
    for key in utility:
        # print(key)
        if key != 'NULL':
            list_key = key.split('_')
            # print(list_key)
            # print(comb(N-1,len(list_key)-1))
            for contributor in list_key:  # contributor: NULL,0,1,2,0_1,0_2,1_2,0_1_2 
                # print('contributor to coalition', contributor, list_key) # print check
                if len(list_key) == 1:
                    marginal_contribution = utility[key] - utility['NULL']
                    pertumation_num[int(contributor)].append(str(key)+'-'+'NULL')
                elif len(list_key) == 2:
                    for i in list_key:
                        if i!=contributor:
                            j = i 
                    marginal_contribution = utility[key] - utility[j]
                    pertumation_num[int(contributor)].append(str(key)+'-'+str(j))
                else:
                    lst = []
                    for i in list_key:
                        if i!=contributor:
                            lst.append(i)
                        str_key = list2str(lst)
                    marginal_contribution = utility[key] - utility[str_key]
                    pertumation_num[int(contributor)].append(str(key)+'-'+str(str_key))
                # print('marginal:', marginal_contribution) # print check
                shapley_dict[int(contributor)] += marginal_contribution /((comb(N-1,len(list_key)-1))*N)
                permutation_value[int(contributor)].append(marginal_contribution /((comb(N-1,len(list_key)-1))*N))
    # print(pertumation_num)
        
        for i in range(len(permutation_key)):
            permutation_dict[permutation_key[i]] = permutation_value[i]      
        # print(permutation_dict)  # {0: [25.0, 12.5, ...], 1: [30.0, 14.166666666666666, ...], ...,3: [20.0, 12.5, 15.0, ...]}
    return shapley_dict, permutation_dict, pertumation_num


def mab_based_cal_shapley_t(utility, N, permutation_num, p_t_list):
    shapley_dict = {}
    weighted_mc_list = []
    # print(permutation_num)
    for i in range(N):
        # print(i)
        mc_list = []
        for j in range(len(permutation_num[i])):
            # print(j)
            # print(permutation_num[i][j])
            permutation = permutation_num[i][j].split('-')
            # print(permutation)
            marginal_contribution = utility[permutation[0]] - utility[permutation[1]]
            # print(marginal_contribution)
            mc_list.append(marginal_contribution)
        # print(mc_list)
        weighted_mc = [x * y for x, y in zip(p_t_list[i], mc_list)]
        weighted_mc_list.append(weighted_mc)
        # print(weighted_mc)
        shapley_dict[i] = sum(weighted_mc)
    return shapley_dict, weighted_mc_list
        # for k in range(para_K):
        #     MC_t[i][a[k]]=weighted_mc[k] / sum(weighted_mc)
        #     # print(MC_t[i][a[k]])
        #     T[i][a[k]]=T[i][a[k]]+1
        #     # print(T[i][a[k]])
        #     SVR[i][a[k]]+=MC_t[i][a[k]]
        #     # print(SVR[i][a[k]])
        #     B[i][a[k]]=-math.sqrt(T[i][a[k]]/para_H)+SVR[i][a[k]]

        # # 选择Top-k个B作为下一轮采样排列的指导
        # B_list = []
        # selected_per_list = []
        # p_t_list = []
        # for i in range(N):
        #     B_list.append(B[i])
        #     a = sorted(heapq.nlargest(para_K, range(len(B[i])), B[i].__getitem__))
        #     selected_per_list.append(a)
        #     # print(a)
        #     p_t = p_t_cal(a, 3)
        #     p_t_list.append(p_t)

def shapley_rank(dict):
    sorted_d = OrderedDict(sorted(dict.items(), key=lambda item: item[1]))
    # print(sorted_d)
    
    result = []
    for i in sorted_d:
        result.append(i)
    # print(result)
    return result


 
if __name__ == '__main__':
    accuracy_dict_mine_five = {'NULL': 0.07, '0': 0.23, '1': 0.19, '2': 0.26, '3': 0.21, '4': 0.12, '0_1': 0.21, '0_2': 0.26, '0_3': 0.26, '0_4': 0.18, '1_2': 0.22, '1_3': 0.24, '1_4': 0.17, '2_3': 0.30, '2_4': 0.20, '3_4': 0.18, '0_1_2': 0.23, '0_1_3': 0.24, '0_1_4': 0.19, '0_2_3': 0.28, '0_2_4': 0.22, '0_3_4': 0.21, '1_2_3': 0.26, '1_2_4': 0.20, '1_3_4': 0.20, '2_3_4': 0.23, '0_1_2_3': 0.25, '0_1_2_4': 0.21, '0_1_3_4': 0.21, '0_2_3_4': 0.23, '1_2_3_4': 0.22, '0_1_2_3_4': 0.23}
    accuracy_dict_mine_four = {'NULL': 0.07, '0': 0.23, '1': 0.19, '2': 0.26, '3': 0.21, '0_1': 0.21, '0_2': 0.26, '0_3': 0.26, '1_2': 0.22, '1_3': 0.24, '2_3': 0.30, '0_1_2': 0.23, '0_1_3': 0.24, '0_2_3': 0.28, '1_2_3': 0.26, '0_1_2_3': 0.25}
    accuracy_dict_four = {'NULL': 0, '0': 100, '1': 120, '2': 50, '3': 80, '0_1': 270, '0_2': 375, '0_3': 250, '1_2': 350, '1_3': 300, '2_3': 200, '0_1_2': 500, '0_1_3': 450, '0_2_3': 400, '1_2_3': 450, '0_1_2_3': 600}


    # shap_dict = {0: 0.019264995803435644, 1: 0.19758999881645045, 2: -0.019276666392882676, 3: 0.136506662145257, 4: 0.04961498553554217}
    # shapley_rank(shap_dict)
    # print(accuracy_dict_mine_four[])

    # accuracy_dict = {(1,): 0.23, (2,): 0.19, (3,): 0.26, (4,): 0.21, (5,): 0.12, (1, 2): 0.21, (1, 3): 0.26, (1, 4): 0.26, (1, 5): 0.18, (2, 3): 0.22, (2, 4): 0.24, (2, 5): 0.17, (3, 4): 0.30, (3, 5): 0.20, (4, 5): 0.18, (1, 2, 3): 0.23, (1, 2, 4): 0.24, (1, 2, 5): 0.19, (1, 3, 4): 0.28, (1, 3, 5): 0.22, (1, 4, 5): 0.21, (2, 3, 4): 0.26, (2, 3, 5): 0.20, (2, 4, 5): 0.20, (3, 4, 5): 0.23, (1, 2, 3, 4): 0.25, (1, 2, 3, 5): 0.21, (1, 2, 4, 5): 0.21, (1, 3, 4, 5): 0.23, (2, 3, 4, 5): 0.22, (1, 2, 3, 4, 5): 0.23, (): 0.07}

    # print(len(accuracy_dict))
    # print(len(accuracy_dict_mine))
    # shapley_dict = shapley(accuracy_dict, 5)
    # shapley_dict_mine_five = cal_shapley(accuracy_dict_mine_five, 5)
    # shapley_dict_mine_four = cal_shapley(accuracy_dict_mine_four, 4)
    # print(shapley_dict)
    # print(shapley_dict_mine_five)
    # print(shapley_dict_mine_four)

    # # 精确计算shapley值
    # shapley_exact = cal_shapley(accuracy_dict_four, 4)
    # print(shapley_exact)

    # delta-shapley值
    iterable = [i for i in range(4)]
    powerset = all_subsets(iterable)
    # print(powerset)
    print(powerset[1:])

    shapley_delta = cal_delta_shapley(accuracy_dict_four, 4, [1, 2, 3], powerset[1:])
    print(shapley_delta)

    # shapley_dict_mab_four = mab_based_cal_shapley(accuracy_dict_four, 4, 2, 100, 3)

    # p_t_cal([1, 3, 4, 7, 8], 4)
    # p_t_cal([0, 1, 2, 3, 4, 7, 9], 5)
    # p_t_cal([6, 4, 3, 7, 1, 2], 4)
    # p_t_cal([7, 6, 3, 2, 8], 4)

    # # p_t_cal函数测试
    # for _ in range(10):
    #     a = [1,2,3,4,5,6,7,8]
    #     b = random.sample(a, random.randint(1,8))
    #     print(b)
    #     p_t_cal(b, 4)

    # str_option = ['0-NULL', '0_2-2']
    # str_option2list(str_option)

    # powerset_t = [[]for _ in range(3)]
    # a = [['0-NULL', '0_1_2-1_2'], ['1-NULL', '0_1_2-0_2'], ['2-NULL', '0_1_2-0_1']]
    # for i in range(3):
    #     powerset_t[i].append(str_option2list(a[i]))

    # N = 5
    # for j in range(N-1):
    #     print(j)
    #     print(comb(N-1,j))

    # shapley_dict_mine_five = cal_stratified_shapley(accuracy_dict_four, 4)
    # shapley_dict_mine_five_stra = cal_shapley(accuracy_dict_four, 4)
    # print(shapley_dict_mine_five_stra)
    # print(shapley_dict_mine_five)

    # shapley_dict, client_stra_importance = cal_mab_stratified_shapley_0(accuracy_dict_four, 4)
    # index = [0, 1]
    # shapley_dict, client_stra_importance = cal_mab_stratified_shapley_t(accuracy_dict_four, 4, index)
    # ratio = copy.deepcopy(client_stra_importance)
    # print(ratio)
    # for i in range(len(client_stra_importance)):
    #     sum_column = sum(client_stra_importance[i])
    #     print(sum_column)
    #     for j in range(len(client_stra_importance[i])):
    #         ratio[i][j] = client_stra_importance[i][j]/sum_column
    # print(ratio)
    # sum_ratio = []
    # for i in range(len(ratio)):
    #     sum_row = 0
    #     for j in range(len(ratio[i])):
    #         sum_row += ratio[j][i]
    #     sum_ratio.append(sum_row)
    # print(sum_ratio)

    # shapley_dict_mine_five = cal_stratified_shapley(accuracy_dict_mine_five, 5)
    # shapley_dict_mine_five_stra = cal_shapley(accuracy_dict_mine_five, 5)
    