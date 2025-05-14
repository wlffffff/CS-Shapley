import numpy as np
from sklearn.cluster import KMeans
import time
import random
import torch
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
# 三种聚类算法都算一下

def K_means_cluster(n_clients,k_means):
    clients = load_clients(n_clients=n_clients)
    a = torch.transpose(torch.tensor(clients),0,1)     
    u,s,v = torch.svd_lowrank(a,q=50) 
    
    print(u.shape)
    print(s.shape)
    print(v.shape)
    a = s[0:30]
    plt.pie(a,autopct='%1.1f%%')
    plt.savefig("a.jpg")
    plt.show()


    initial = random.sample(range(0, n_clients), k_means)  
    init_model = [clients[i] for i in initial]
    indexes2 = [[] for i in range(k_means)] 
    print("init_model:",initial)
    print("shape:",np.array(clients).shape)
    num = 1

    while True:
        clusters = [[] for i in range(k_means)]   
        indexes = [[] for i in range(k_means)]      

        for i in range(n_clients):
      
            distance = []
            for j in range(k_means):
                a = np.sqrt(np.sum(np.square(np.array(clients[i])-np.array(init_model[j]))))
                distance.append(a)
            id = distance.index(min(distance))
            clusters[id].append(clients[i])
            indexes[id].append(i)


        print(np.array(clusters).shape)
        for i in range(k_means):
            print(i)
            a = np.array(clusters[i][0])*0.0
            for j in clusters[i]:
                a = np.array(a) + np.array(j)
            a = np.array(a) / len(clusters[i])
            init_model[i] = a.tolist()

        print(num, indexes, indexes2,'\n')
        if (np.array(indexes).shape == np.array(indexes2).shape) and (np.array(indexes) == np.array(indexes2)).all():
            if num % 100 == 0:
                print("无法收敛，返回")
                return indexes
                
            break
        else:
            num = num + 1
            indexes2 = copy.deepcopy(indexes)

    return indexes

def Kmeans(n_clients):
    clients = load_clients(n_clients=n_clients)
    u,s,v = torch.svd_lowrank(torch.tensor(clients),q=100)            
    
    print(u.shape)
    print(s)
    print(v.shape)
    a = s[0:30]
    plt.pie(a,autopct='%1.1f%%')
    plt.savefig("a.jpg")
    plt.show()

    for k in range(len(s)):             
        if torch.sum(s[0:k])/torch.sum(s) >= 0.9:
            break
    print(k)               

    result = KMeans(k,max_iter=100).fit(clients).labels_    # kmeans++
    print(result)
    cluster = [[] for _ in range(k)]
    for i,index in enumerate(result):
        cluster[index].append(i)

    return cluster


def Kmeans_plusplus(gradients,n_clients,device,epoch):
    clients = load_clients(gradient_locals=gradients)
    clients = torch.tensor(clients).to(device)

    distance = Distance_matrix(n_clients, clients)      
    clients = torch.tensor(distance)

    k = SVD_Dis(n_clients, clients, epoch)

    initial_model = [clients[0]]     
    # print(initial_model[0])
    while len(initial_model) < k:
        distances = []              
        for c in clients:
            min_dis = 100000       
            for i in initial_model:
                dis = torch.norm(i-c,p=2,dim=0).item()
                if dis < min_dis:
                    min_dis = dis  
            distances.append(min_dis)
        # print(distances)
        max_index = distances.index(max(distances))  
        distances[max_index] = 0       

        max_index = distances.index(max(distances))  
        initial_model.append(clients[max_index])

    num = 1
    indexes2 = [[] for i in range(k)]  
    while True:
        clusters = [[] for _ in range(k)]    
        indexes = [[] for _ in range(k)]     

        for i in range(n_clients):
            
            distance = []
            for j in range(k):
                # a = np.sqrt(np.sum(np.square(np.array(clients[i])-np.array(initial_model[j]))))
                a = torch.norm(clients[i]-initial_model[j],p=2,dim=0).item()
                distance.append(a)
            id = distance.index(min(distance))
            clusters[id].append(clients[i])
            indexes[id].append(i)

        
        for i in range(k):
            a = clusters[i][0]*0.0
            for j in clusters[i]:
                a += j
            a = torch.div(a,len(clusters[i]))
            initial_model[i] = a
            
        if (np.array(indexes).shape == np.array(indexes2).shape) and (np.array(indexes) == np.array(indexes2)).all():
            if num % 100 == 0:
                print("无法收敛，返回")
                return indexes
            break
        else:
            num = num + 1
            indexes2 = copy.deepcopy(indexes)

    return indexes

# load model and get para
def load_clients(gradient_locals):
    model_states = [[] for _ in range(len(gradient_locals))]
    # print(model_states)
    for i in range(len(gradient_locals)):
        grad_model = gradient_locals[i]
        # print(grad_model)   

        for name in grad_model.keys():
            a = grad_model[name].view(-1).tolist()
            model_states[i].extend(a)

    return model_states

# compute distance matrix
def Distance_matrix(n_clients,clients):
    distance = []  
    for i in range(n_clients):
        dis = []
        for j in range(n_clients):
            d = torch.norm(torch.tensor(clients[i]) - torch.tensor(clients[j]), p=2, dim=0).item()  
            if i == j:
                d = 1000
            dis.append(d)
        dis[i] = min(dis)
        for c in range(n_clients):
            dis[c] = (max(dis) - dis[c]) / (max(dis) - min(dis))  
        distance.append(dis)
    return distance

# def flatten(source):
#     return torch.cat([value.flatten() for value in source.values()])

def pairwise_cossim(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        # print(source1)
        for j, source2 in enumerate(sources):
            s1 = source1
            s2 = source2
            angles[i,j] = torch.sum(s1*s2)/(torch.norm(s1)*torch.norm(s2)+1e-12)
    return angles.numpy()

def cosine_sim(gradient_i, gradient_j):
    for name in gradient_i.keys():
        grad_i = gradient_i[name].view(-1).tolist()
    # print(torch.norm(torch.tensor(grad_i)))
    for name in gradient_j.keys():
        grad_j = gradient_j[name].view(-1).tolist()
    cosine = (torch.norm(torch.tensor(grad_i))*torch.norm(torch.tensor(grad_j))+1e-12)
    # print(float(cosine))
    return float(cosine)

def compute_pairwise_similarities(clients):
    return pairwise_cossim([client for client in clients])

# get cluster number by using SVD
def SVD_Dis(n_clients,clients,epoch):

    a = torch.transpose(torch.tensor(clients), 0, 1)  
    u, s, v = torch.svd_lowrank(a, q=n_clients)  

    print(u.shape)
    print(s)
    print(v.shape)
    plt.pie(s, autopct='%1.1f%%')
    plt.savefig("./result/picture/a_{}.jpg".format(epoch))
    plt.show()
    k = 0
    for k in range(len(s)):  
        if torch.sum(s[0:k]) / torch.sum(s) >= 0.8:
            break
    print(k)  

    return k



def k_means(gradients,n_clients,n_cluster,device):
    clients = load_clients(gradient_locals=gradients)
    clients = torch.tensor(clients).to(device)

    similarities = compute_pairwise_similarities(clients)
    # print(similarities)
    # distance = Distance_matrix(10, clients)
    # print(distance)
    kmeans = KMeans(n_cluster)
    clusters = kmeans.fit_predict(similarities)
    # print(clusters)
    cluster_result = [[] for _ in range(n_cluster)]
    # print(cluster_result)
    cluster_id = [i for i in range(n_cluster)]
    # print(cluster_id)
    for i in range(n_clients):
        for id in cluster_id:
            if clusters[i] == id:
                cluster_result[id].append(i)
    # print(cluster_result)
    return cluster_result

def orderdict_zeros_like(orderdict):
    ms = copy.deepcopy(orderdict)
    for key, value in orderdict.items():
        ms[key] = torch.zeros_like(value)
    return ms

def cluster_agg(gradient_list):
    result = orderdict_zeros_like(gradient_list[0])
    for i in gradient_list:
        for key, value in i.items():
            if key in result:
                result[key] += value
            else:
                result[key] =value
    if len(gradient_list) == 0:
        return False
    final_result = copy.deepcopy(result)
    for key, value in result.items():
        result[key] = value / len(gradient_list)
    return final_result

def cluster_shapley_average_distribute(num_users, cluster_shapley_dict, cluster_result):
    client_shapley_dict = {}
    for i in range(num_users):
        id = -1
        for j in cluster_result:
            id += 1
            if i in j:
                client_shapley_dict[i] = cluster_shapley_dict[id] / len(cluster_result[id])
    print(client_shapley_dict)
    return client_shapley_dict

def cluster_shapley_similarity_distribute(num_users, cluster_shapley_dict, cluster_result, cluster_represent_gradient, gradient_locals):
    client_shapley_dict = {}
    for i in range(num_users):
        id = -1
        for j in cluster_result:
            id += 1
            if i in j:
                client_shapley_dict[i] = cluster_shapley_dict[id] * cosine_sim(gradient_locals[i], cluster_represent_gradient[id])
    print(client_shapley_dict)
    return client_shapley_dict

if __name__ == "__main__":
    # global_nums = 30
    gpu = -1
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    # clients_id = Kmeans_plusplus(n_clients=global_nums,device=device,epoch='N') # 使用kmeans++进行聚类
    # print(clients_id)

    # grad_model = torch.load(f'../cache/client_model_2.pt')
    # print(grad_model)
    # print(len(grad_model))
    # for name in grad_model.keys():
    #     a = grad_model[name].view(-1).tolist()
    #     # print("!!!!!!!!!!!!!!!!!!!!")
    #     # print(a)

    # n_clients = 10
    # n_clusters=3
    # gpu = -1
    # device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

    # model_states = [[] for _ in range(n_clients)]      
    # for i in range(n_clients):
    #     grad_model = torch.load('../cache/gradient_{}.pt'.format(i))       
        
    #     for name in grad_model.keys():
    #         a = grad_model[name].view(-1).tolist()
    #         model_states[i].extend(a)

    # clients = model_states
    # clients = torch.tensor(clients).to(device)

    # similarities = compute_pairwise_similarities(clients)
    # # print(similarities)
    # distance = Distance_matrix(10, clients)
    # # print(distance)
    # kmeans = KMeans(n_clusters=3)
    # clusters = kmeans.fit_predict(similarities)
    # print(clusters)
    # cluster_result = [[] for _ in range(n_clusters)]
    # print(cluster_result)
    # cluster_id = [i for i in range(n_clusters)]
    # print(cluster_id)
    # for i in range(n_clients):
    #     for id in cluster_id:
    #         if clusters[i] == id:
    #             cluster_result[id].append(i)
    # print(cluster_result)


################
    # gradient_i = torch.load('../cache/gradient_0.pt')
    # gradient_j = torch.load('../cache/gradient_1.pt')
    # print(gradient_j)
    # # gradient_i = gradient_i.to(device)
    # # gradient_j = gradient_j.to(device)
    # for name in gradient_i.keys():
    #     grad_i = gradient_i[name].view(-1).tolist()
    # print(torch.norm(torch.tensor(grad_i)))
    # for name in gradient_j.keys():
    #     grad_j = gradient_j[name].view(-1).tolist()
    # print((torch.norm(torch.tensor(grad_i))*torch.norm(torch.tensor(grad_j))+1e-12))

    # print(torch.norm(torch.tensor(gradient_i)))

    # gradient_i_value = []
    # gradient_j_value = []
    # for key_i, value_i in gradient_i.items():
    #     # print(value_i)
    #     gradient_i_value.append(value_i)
    # # print(gradient_i_value)
    # gradient_i_value_cpu = [i.cpu() for i in gradient_i_value]
    # grad_i = np.array(gradient_i_value_cpu).flatten()
    # # print(grad_i)
    # for key_j, value_j in gradient_j.items():
    #     gradient_j_value.append(value_j)
    # gradient_j_value_cpu = [j.cpu() for j in gradient_j_value]
    # grad_j = np.array(gradient_j_value_cpu).flatten()
    # print(grad_j)
    # similarity = cosine_similarity(grad_i.unsqueeze(0), grad_j.unsqueeze(0))[0][0]
    # angle = torch.sum(torch.tensor(grad_i)*torch.tensor(grad_j))/(torch.norm(torch.tensor(grad_i))*torch.norm(torch.tensor(grad_j))+1e-12)
    # print(angle)

    a = torch.tensor(0.0379)
    print(a)
    a_float = float(a)
    print(a_float)





