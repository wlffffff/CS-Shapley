import os
import numpy as np
import copy
import random
import torch
from torchvision import datasets, transforms
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Method     : {args.method}')
    print(f'    Dataset     : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    Mode     : {args.data_quality_mode}')
    print(f'    Num of users     : {args.num_users}')
    # print(f'    Learning rate  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    # print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    if args.method == "mab_n_stratified_shapley" or args.method == "mab_mc_shapley" :
        print(f'    Para_H        : {args.para_H}')
        print(f'    Para_K        : {args.para_K}')
    if args.method == "mab_mc_shapley" :
        print(f'    Iteration        : {args.iteration}')
    # print(f'    Fraction of users  : {args.frac}')
    # print(f'    Local Batch size   : {args.bs_train}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

def worst_fraction(acc_list, fraction):
    acc_sort = sorted(acc_list, reverse=False)
    worst = [acc_sort[i] for i in range(int(len(acc_list)*fraction))]
    return sum(worst)/len(worst)

def best_fraction(acc_list, fraction):
    acc_sort = sorted(acc_list, reverse=True)
    best = [acc_sort[i] for i in range(int(len(acc_list)*fraction))]
    return sum(best)/len(best)

def add_noise(args, y_train, dict_users):
    np.random.seed(args.seed)

    gamma_s = [1]*int(args.level_n_system * args.num_users) +[0]*int((1-args.level_n_system) * args.num_users)# np.random.binomial(1, args.level_n_system, args.num_users)
    np.random.shuffle(gamma_s)
    gamma_c_initial = np.random.rand(args.num_users)
    gamma_c_initial = (1 - args.level_n_lowerb) * gamma_c_initial + args.level_n_lowerb
    gamma_c = gamma_s * gamma_c_initial

    y_train_noisy = copy.deepcopy(y_train)

    real_noise_level = np.zeros(args.num_users)
    for i in np.where(gamma_c > 0)[0]:
        sample_idx = np.array(list(dict_users[str(i)]))
        prob = np.random.rand(len(sample_idx))
        noisy_idx = np.where(prob <= gamma_c[i])[0]
        y_train_noisy[sample_idx[noisy_idx]] = np.random.randint(0, 10, len(noisy_idx))
        noise_ratio = np.mean(y_train[sample_idx] != y_train_noisy[sample_idx])
        print("Client %d, noise level: %.4f (%.4f), real noise ratio: %.4f" % (i, gamma_c[i], gamma_c[i] * 0.9, noise_ratio))
        real_noise_level[i] = noise_ratio
    return (y_train_noisy, gamma_s, real_noise_level)


def get_original_transforms(dataset):
    """获取数据集的原始transform列表"""
    if hasattr(dataset.transform, 'transforms'):
        return dataset.transform.transforms
    elif dataset.transform is not None:
        return [dataset.transform]
    else:
        return []
    
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        # logger.debug(tensor)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


# def adjust_quality(dataset_train, dict_users, args):
#     """
#     调整数据集质量和数量
#     :param dataset_train: 原始训练数据集
#     :param dict_users: 用户数据索引字典
#     :param args: 命令行参数
#     :return: 处理后的用户数据索引字典
#     """
#     if args.data_quality_mode == 'clean':
#         return dict_users, dataset_train  # 不做任何处理
    
#     modified_dataset = copy.deepcopy(dataset_train)

#     # 确保所有索引都是Python原生整数
#     dict_users = {k: [int(i) for i in v] for k, v in dict_users.items()}

#     # 初始化客户端特定transform字典
#     client_transforms = {}

#     # 创建每个用户的数据集子集
#     user_datasets = []
#     for user in range(args.num_users):
#         indices = dict_users[user]
#         subset = torch.utils.data.Subset(dataset_train, indices)
#         user_datasets.append(subset)
    
#     if args.data_quality_mode == 'noise':
#         # 为每个客户端添加不同级别的高斯噪声
#         noise_mean_levels = [x * 0.01 for x in range(args.num_users)]
#         noise_std_levels = [((x / 32)) for x in range(args.num_users)]
#         client_transforms = {}
#         for i in range(args.num_users):
#             client_transforms[i] = transforms.Compose([
#                 *get_original_transforms(modified_dataset),
#                 transforms.Resize((32, 32)),
#                 transforms.RandomCrop(32, padding=4, padding_mode="reflect", fill=0),
#                 # transforms.RandomRotation(15),
#                 # transforms.RandomHorizontalFlip(),
#                 AddGaussianNoise(noise_mean_levels[i], noise_std_levels[i]),
                
#                 # transforms.ToTensor(),
#             ])
#             print(i)
#             print(client_transforms[i])
    
#     elif args.data_quality_mode == 'blur':
#         # 为每个客户端添加不同级别的模糊
#         blur_kernel_size = [x * 2 + 1 for x in range(args.num_users)]
#         blur_std = [((1 + 0.4 * x)) for x in range(args.num_users)]
#         client_transforms = {}
#         for i in range(args.num_users):
#             client_transforms[i] = transforms.Compose([
#                 *get_original_transforms(modified_dataset),
#                 transforms.Resize((32, 32)),
#                 transforms.RandomCrop(32, padding=4, padding_mode="reflect", fill=0),
#                 # transforms.RandomRotation(15),
#                 # transforms.RandomHorizontalFlip(),
#                 transforms.GaussianBlur(blur_kernel_size[i], blur_std[i]),
                
#                 # transforms.ToTensor(),
#             ])
#             print(i)
#             print(client_transforms[i])
    
#     elif args.data_quality_mode == 'count':
#         # 调整每个客户端的数据量
#         total_samples = len(dataset_train)
#         split_lengths = []
        
#         for i in range(args.num_users):
#             # 每个客户端的数据量递减
#             split_lengths.append(int((1 - 0.1 * i) * (total_samples / args.num_users)))
        
#         # 确保总和不超过总样本数
#         split_lengths = [int(x * total_samples / sum(split_lengths)) for x in split_lengths]
#         remaining = total_samples - sum(split_lengths)
#         split_lengths[0] += remaining  # 将剩余样本添加到第一个客户端
        
#         # 重新分配数据
#         indices = torch.randperm(total_samples).tolist()
#         new_dict_users = {}
#         start_idx = 0
        
#         for i in range(args.num_users):
#             end_idx = start_idx + split_lengths[i]
#             new_dict_users[i] = indices[start_idx:end_idx]
#             start_idx = end_idx
        
#         return new_dict_users
    
#     elif args.data_quality_mode == 'mask':
#         # 为每个客户端添加不同程度的随机擦除
#         erase_params = [
#             # (min(1 * i / args.num_users, 0.9), min(1.5 * i / args.num_users, 0.95), 0.3, 3.3)  
#             (min(0.1 * i, 0.9), min(0.15 * i, 0.95), 0.3, 3.3)  
#             for i in range(args.num_users)
#         ]
        
#         for i in range(args.num_users):
#             client_transforms[i] = transforms.Compose([
#                 *get_original_transforms(modified_dataset),
#                 transforms.Resize((32, 32)),
#                 transforms.RandomCrop(32, padding=4, padding_mode="reflect", fill=0),
#                 # transforms.RandomRotation(15),
#                 # transforms.RandomHorizontalFlip(),
#                 transforms.RandomErasing(p=1.0, scale=(erase_params[i][0], erase_params[i][1]),ratio=(erase_params[i][2], erase_params[i][3])),
                
#                 # transforms.ToTensor(),
#             ])
#             print(i)
#             print(client_transforms[i])
    
#     elif args.data_quality_mode == 'flip':
#         # 为每个客户端添加不同程度的标签翻转
#         for i in range(args.num_users):
#             user_datasets[i].dataset = deepcopy(user_datasets[i].dataset)
#             num_samples = len(user_datasets[i])
#             wrong_labels = int(0.1 * i * num_samples)
            
#             # 获取原始标签
#             targets = user_datasets[i].dataset.targets
#             if isinstance(targets, list):
#                 targets = torch.tensor(targets)
            
#             # 随机选择要翻转的标签
#             flip_indices = random.sample(range(num_samples), wrong_labels)
#             for idx in flip_indices:
#                 original_idx = user_datasets[i].indices[idx]
#                 targets[original_idx] = random.randint(0, args.num_classes - 1)
            
#             user_datasets[i].dataset.targets = targets

#     # 创建每个用户的数据集子集并应用transform
#     user_datasets = {}
#     for i in range(args.num_users):
#         indices = dict_users[i]
#         subset = torch.utils.data.Subset(modified_dataset, indices)

#         # 获取第一个样本（索引 0 对应原始 indices 中的第一个值 1）
#         img, label = subset[0]

#         # 打印图像和标签信息
#         print("Image shape:", img.shape)  # 应为 torch.Size([3, H, W])
#         print("Image dtype:", img.dtype)  # 应为 torch.float32
#         print("Image min/max:", img.min().item(), img.max().item())  # 标准化后范围应为 ~[-1, 1]
#         print("Label:", label)  # 整数，范围 0-9
        
#         if args.data_quality_mode in ['noise', 'blur', 'mask']:
#             subset.dataset.transform = client_transforms[i]
        
#         user_datasets[i] = subset

#     if args.visualize:
#         visualize_processed_data(user_datasets, args)
#         if args.data_quality_mode in ['noise', 'blur', 'mask']:
#             compare_samples(dataset_train, user_datasets, args)
#     print(user_datasets)
#     return dict_users, user_datasets


# transform
def adjust_quality(dataset_train, dict_users, args):
    """
    调整数据集质量和数量
    :param dataset_train: 原始训练数据集
    :param dict_users: 用户数据索引字典
    :param args: 命令行参数
    :return: 
        - dict_users: 处理后的用户数据索引字典
        - modified_dataset: 修改后的完整数据集(格式与dataset_train相同)
    """
    
    # 深拷贝原始数据集
    modified_dataset = copy.deepcopy(dataset_train)
    
    # 确保所有索引都是Python原生整数
    dict_users = {int(k): [int(i) for i in v] for k, v in dict_users.items()}

    # 初始化客户端特定transform字典
    client_transforms = {}


    # 创建transform配置 (保持原有代码不变)
    if args.data_quality_mode == 'clean':
        for i in range(args.num_users):
            client_transforms[i] = transforms.Compose([
                *get_original_transforms(modified_dataset),
                # transforms.Resize((32, 32)),
                # transforms.RandomCrop(32, padding=4, padding_mode="reflect", fill=0),
                # transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    elif args.data_quality_mode == 'noise':
        # 为每个客户端添加不同级别的高斯噪声
        noise_mean_levels = [x * 0.01 for x in range(args.num_users)]
        noise_std_levels = [((x / 32)) for x in range(args.num_users)]
        for i in range(args.num_users):
            client_transforms[i] = transforms.Compose([
                *get_original_transforms(modified_dataset),
                # transforms.Resize((32, 32)),
                # transforms.RandomCrop(32, padding=4, padding_mode="reflect", fill=0),
                # transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                AddGaussianNoise(noise_mean_levels[i], noise_std_levels[i]),
            ])
            # print(i)
            # print(client_transforms[i])

    elif args.data_quality_mode == 'blur':
        blur_kernel_size = [x * 2 + 1 for x in range(args.num_users)]
        blur_std = [((1 + 0.4 * x)) for x in range(args.num_users)]
        for i in range(args.num_users):
            client_transforms[i] = transforms.Compose([
                *get_original_transforms(modified_dataset),
                # transforms.Resize((32, 32)),
                # transforms.RandomCrop(32, padding=4, padding_mode="reflect", fill=0),
                # transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.GaussianBlur(blur_kernel_size[i], blur_std[i]),
            ])
            # print(i)
            # print(client_transforms[i])

    elif args.data_quality_mode == 'mask':
        # 为每个客户端添加不同程度的随机擦除
        erase_params = [
            # (min(1 * i / args.num_users, 0.9), min(1.5 * i / args.num_users, 0.95), 0.3, 3.3)  
            (min(0.01667 * i, 0.9), min(0.025 * i, 0.95), 0.3, 3.3)  
            for i in range(args.num_users)
        ]
        for i in range(args.num_users):
            client_transforms[i] = transforms.Compose([
                *get_original_transforms(modified_dataset),
                # transforms.Resize((32, 32)),
                # transforms.RandomCrop(32, padding=4, padding_mode="reflect", fill=0),
                # transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomErasing(p=1.0, scale=(erase_params[i][0], erase_params[i][1]), ratio=(erase_params[i][2], erase_params[i][3])),
            ])
            # print(i)
            # print(client_transforms[i])

    elif args.data_quality_mode == 'flip':
        # 标签翻转直接修改数据集
        if isinstance(modified_dataset.targets, list):
            modified_dataset.targets = torch.tensor(modified_dataset.targets)
        for i in range(args.num_users):
            indices = dict_users[i]
            flip_count = int(0.1 * i * len(indices))
            flip_indices = random.sample(indices, flip_count)
            for idx in flip_indices:
                modified_dataset.targets[idx] = random.randint(0, args.num_classes - 1)

    elif args.data_quality_mode == 'count':
        # 数量模式保持原有逻辑
        total_samples = len(dataset_train)
        split_lengths = []
        for i in range(args.num_users):
            # 每个客户端的数据量递减
            split_lengths.append(int((1 - 0.5 * i * (total_samples / args.num_users))))
        
        # 确保总和不超过总样本数
        split_lengths = [int(x * total_samples / sum(split_lengths)) for x in split_lengths]
        remaining = total_samples - sum(split_lengths)
        split_lengths[0] += remaining # 将剩余样本添加到第一个客户端
        
        # 重新分配数据
        indices = torch.randperm(total_samples).tolist()
        new_dict_users = {}
        start_idx = 0

        for i in range(args.num_users):
            end_idx = start_idx + split_lengths[i]
            new_dict_users[i] = indices[start_idx:end_idx]
            start_idx = end_idx
        return new_dict_users, modified_dataset

    # 创建TransformApplier类来应用并缓存transform结果
    # class TransformApplier:
    #     def __init__(self, dataset, transform):
    #         self.dataset = dataset
    #         self.transform = transform
    #         self.cache = {}
            
    #     def __getitem__(self, idx):
    #         if idx not in self.cache:
    #             img, label = self.dataset[idx]
    #             print(img.dtype)
    #             if self.transform is not None:
    #                 # 确保图像是PIL Image或numpy数组
    #                 if isinstance(img, torch.Tensor):
    #                     img = transforms.ToPILImage()(img)
    #                 img = self.transform(img)
    #             self.cache[idx] = (img, label)
    #         return self.cache[idx]
        
    #     def __len__(self):
    #         return len(self.dataset)
    

    class TransformApplier:
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
            self.cache = {}
            
        def __getitem__(self, idx):
            if idx not in self.cache:
                img, label = self.dataset[idx]
                # 统一转换为 ndarray 格式 (确保是 HWC 格式)
                if isinstance(img, torch.Tensor):
                    # print("Tensor")
                    # img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
                    img = img.permute(1, 2, 0)
                    img = (img * 255).clamp(0, 255).byte().numpy()
                    # # 确保是 float32 类型并标准化到 [-1,1]
                    # img = (img - 0.5) / 0.5  # 标准化到 [-1, 1]

                    # print("Image shape:", img.shape)  # 应为 torch.Size([3, H, W])
                    # print("Image dtype:", img.dtype)  # 应为torch.float32
                    # print("Image min/max:", img.min().item(), img.max().item())  # 标准化后范围应为 ~[-1, 1]
                    # print("Label:", label)  # 整数，范围 0-9
                elif isinstance(img, np.ndarray):
                    # 如果是 ndarray，确保是 HWC 格式
                    if img.shape[0] == 3:  # 如果是 (3, H, W)，转换为 (H, W, 3)
                        img = img.transpose(1, 2, 0)  # CHW -> HWC
                else:  # PIL Image
                    # 对于 PIL 图像，先转换为 ndarray（HWC 格式）
                    img = np.array(img)
                
                #  # 标准化到 [-1,1]（如果原始范围是 [0,1]）
                # img = (img - 0.5) / 0.5


            # # 打印图像和标签信息
            # print("Image shape:", img.shape)  # 应为 torch.Size([3, H, W])
            # print("Image dtype:", img.dtype)  # 应为 torch.float32
            # print("Image min/max:", img.min().item(), img.max().item())  # 标准化后范围应为 ~[-1, 1]
            # print("Label:", label)  # 整数，范围 0-9

            if self.transform is not None:
                # print(self.transform)
                # print(type(img))
                img = self.transform(img)  # 应用变换（保持 float32）
            
            self.cache[idx] = (img, label)
            return self.cache[idx]
        
        def __len__(self):
            return len(self.dataset)


    # 应用transform并修改数据集
    for client_id in range(args.num_users):
        indices = dict_users[client_id]
        transform = client_transforms[client_id]
        
        # 创建transform应用器
        applier = TransformApplier(modified_dataset, transform)

        # img, label = applier[0]
        # print(img.shape, img.dtype, img.min(), img.max())  # 检查范围和类型
        
        # 应用transform并更新数据集
        for idx in indices:
            modified_img, label = applier[idx]

            if isinstance(modified_img, torch.Tensor):
                modified_img = (modified_img + 1) * 127.5  # (-1,1) -> (0,255)
                modified_img = modified_img.byte()  # 转换为 uint8

            # 对于 numpy 数组
            if isinstance(modified_img, np.ndarray):
                modified_img = (modified_img + 1) * 127.5
                modified_img = modified_img.astype(np.uint8)
            # print(modified_img)
            
            # 更新数据集
            modified_dataset.data[idx] = np.array(modified_img.permute(1, 2, 0))
            # if hasattr(modified_dataset, 'data'):
            #     # 处理不同图像格式 (CHW vs HWC)
            #     if isinstance(modified_img, torch.Tensor):
            #         if modified_dataset.data.shape[3] == 3:  # 原始是HWC
            #             modified_dataset.data[idx] = modified_img.permute(1, 2, 0).numpy()
            #         else:  # 原始是CHW
            #             modified_dataset.data[idx] = modified_img.numpy()
            #     else:  # PIL Image
            #         modified_dataset.data[idx] = np.array(modified_img)
            # else:
            #     modified_dataset.samples[idx] = (modified_img, label)

    # # 可视化处理后的数据
    # if args.visualize:
    #     visualize_processed_data_full(modified_dataset, dict_users, args)
    #     if args.data_quality_mode in ['noise', 'blur', 'mask']:
    #         compare_samples_full(dataset_train, modified_dataset, dict_users, args)

    return dict_users, modified_dataset


# # new_version
# def adjust_quality(dataset_train, dict_users, args):
#     """
#     调整数据集质量和数量
#     :param dataset_train: 原始训练数据集
#     :param dict_users: 用户数据索引字典
#     :param args: 命令行参数
#     :return: 
#         - dict_users: 处理后的用户数据索引字典
#         - modified_dataset: 修改后的完整数据集(格式与dataset_train相同)
#     """
#     if args.data_quality_mode == 'clean':
#         return dict_users, dataset_train  # 不做任何处理
    
#     # 深拷贝原始数据集
#     modified_dataset = copy.deepcopy(dataset_train)
    
#     # 确保所有索引都是Python原生整数
#     dict_users = {k: [int(i) for i in v] for k, v in dict_users.items()}

#     if args.data_quality_mode == 'noise':
#         # 为每个客户端添加不同级别的高斯噪声
#         noise_mean_levels = [x * 0.01 for x in range(args.num_users)]
#         noise_std_levels = [(x / 32) for x in range(args.num_users)]
        
#         for client_id in range(args.num_users):
#             indices = dict_users[client_id]
#             for idx in indices:
#                 # 获取原始图像
#                 img, label = modified_dataset[idx]
                
#                 # 添加噪声
#                 if isinstance(img, torch.Tensor):
#                     noise = torch.randn_like(img) * noise_std_levels[client_id] + noise_mean_levels[client_id]
#                     modified_img = torch.clamp(img + noise, 0, 1)
#                 else:  # 如果是PIL Image或numpy数组
#                     img_tensor = transforms.ToTensor()(img)
#                     noise = torch.randn_like(img_tensor) * noise_std_levels[client_id] + noise_mean_levels[client_id]
#                     modified_img = transforms.ToPILImage()(torch.clamp(img_tensor + noise, 0, 1))
                
#                 # 更新数据集
#                 if hasattr(modified_dataset, 'data'):  # CIFAR等数据集
#                     modified_dataset.data[idx] = modified_img
#                 else:  # 其他类型数据集
#                     modified_dataset.samples[idx] = (modified_img, label)

#     elif args.data_quality_mode == 'blur':
#         # 为每个客户端添加不同级别的模糊
#         blur_kernel_sizes = [x * 2 + 1 for x in range(args.num_users)]
#         blur_stds = [(1 + 0.4 * x) for x in range(args.num_users)]
        
#         for client_id in range(args.num_users):
#             indices = dict_users[client_id]
#             blur_transform = transforms.GaussianBlur(
#                 kernel_size=blur_kernel_sizes[client_id],
#                 sigma=blur_stds[client_id]
#             )
            
#             for idx in indices:
#                 img, label = modified_dataset[idx]
                
#                 # 应用模糊
#                 if isinstance(img, torch.Tensor):
#                     modified_img = blur_transform(img.unsqueeze(0)).squeeze(0)
#                 else:
#                     img_tensor = transforms.ToTensor()(img).unsqueeze(0)
#                     modified_img = transforms.ToPILImage()(blur_transform(img_tensor).squeeze(0))
                
#                 # 更新数据集
#                 if hasattr(modified_dataset, 'data'):
#                     modified_dataset.data[idx] = modified_img
#                 else:
#                     modified_dataset.samples[idx] = (modified_img, label)

#     elif args.data_quality_mode == 'mask':
#         # 为每个客户端添加不同程度的随机擦除
#         erase_scales = [(0.0 + 0.8 * i / args.num_users) for i in range(args.num_users)]
        
#         for client_id in range(args.num_users):
#             indices = dict_users[client_id]
            
#             for idx in indices:
#                 img, label = modified_dataset[idx]
                
#                 # 转换为tensor进行处理
#                 if not isinstance(img, torch.Tensor):
#                     img_tensor = transforms.ToTensor()(img).unsqueeze(0)
#                 else:
#                     if img.dim() == 3 and img.size(0) == 3:  # CHW
#                         img_tensor = img.unsqueeze(0)
#                     else:
#                         img_tensor = img.permute(2, 0, 1).unsqueeze(0) if img.dim() == 3 else img.unsqueeze(0).unsqueeze(0)
                    
                
#                 # 应用随机擦除
#                 erase_transform = transforms.RandomErasing(
#                     p=1.0,
#                     scale=(erase_scales[client_id], erase_scales[client_id] + 0.1),
#                     ratio=(0.3, 3.3)
#                 )
#                 modified_img_tensor = erase_transform(img_tensor).squeeze(0)
                
#                 # 更新数据集
#                 if hasattr(modified_dataset, 'data'):
#                     if modified_dataset.data.shape[3] == 3:  # NHWC
#                         modified_dataset.data[idx] = modified_img_tensor.permute(1, 2, 0).numpy()
#                     else:  # NCHW
#                         modified_dataset.data[idx] = modified_img_tensor.numpy()
#                 else:
#                     modified_img_pil = transforms.ToPILImage()(modified_img_tensor)
#                     modified_dataset.samples[idx] = (modified_img_pil, label)

#     elif args.data_quality_mode == 'flip':
#         # 为每个客户端添加不同程度的标签翻转
#         if isinstance(modified_dataset.targets, list):
#             modified_dataset.targets = torch.tensor(modified_dataset.targets)
        
#         for client_id in range(args.num_users):
#             indices = dict_users[client_id]
#             flip_count = int(0.1 * client_id * len(indices))
#             flip_indices = random.sample(indices, flip_count)
            
#             for idx in flip_indices:
#                 modified_dataset.targets[idx] = random.randint(0, args.num_classes - 1)

#     elif args.data_quality_mode == 'count':
#         # 数量模式需要返回新的dict_users
#         total_samples = len(dataset_train)
#         split_lengths = [int((1 - 0.1 * i) * (total_samples / args.num_users)) for i in range(args.num_users)]
#         split_lengths = [int(x * total_samples / sum(split_lengths)) for x in split_lengths]
#         remaining = total_samples - sum(split_lengths)
#         split_lengths[0] += remaining
        
#         indices = torch.randperm(total_samples).tolist()
#         new_dict_users = {}
#         start_idx = 0
        
#         for i in range(args.num_users):
#             end_idx = start_idx + split_lengths[i]
#             new_dict_users[i] = [int(x) for x in indices[start_idx:end_idx]]
#             start_idx = end_idx
        
#         return new_dict_users, modified_dataset

#     # # 可视化处理后的数据
#     # if args.visualize:
#     #     # 需要更新可视化函数以使用完整数据集和dict_users
#     #     visualize_processed_data_full(modified_dataset, dict_users, args)
#     #     if args.data_quality_mode in ['noise', 'blur', 'mask']:
#     #         compare_samples_full(dataset_train, modified_dataset, dict_users, args)

#     return dict_users, modified_dataset

def visualize_processed_data(user_datasets, args):
    """可视化处理后的数据"""
    print("\nVisualizing processed data samples...")
    
    # 可视化前几个客户端的样本
    for client_id in range(min(5, args.num_users)):
        subset = user_datasets[client_id]
        visualize_client_data(subset, range(len(subset)), client_id, args, title_suffix=f"({args.data_quality_mode})")

def compare_samples(original_dataset, user_datasets, args):
    """对比显示原始和处理后的样本"""
    client_id = 0  # 使用第一个客户端做对比
    subset = user_datasets[client_id]
    
    # 随机选择一些样本
    sample_indices = random.sample(range(len(subset)), min(5, len(subset)))
    
    # 获取原始和处理后的样本
    original_samples = []
    processed_samples = []
    labels = []
    
    for idx in sample_indices:
        original_idx = subset.indices[idx]
        original_samples.append(original_dataset[original_idx][0])
        processed_samples.append(subset[idx][0])
        labels.append(subset[idx][1])
    
    # 创建图像网格
    original_grid = make_grid(original_samples, nrow=5, normalize=True)
    processed_grid = make_grid(processed_samples, nrow=5, normalize=True)
    
    # 显示对比图
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_grid.permute(1, 2, 0).numpy())
    plt.title("Original Samples")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed_grid.permute(1, 2, 0).numpy())
    plt.title(f"Processed Samples ({args.data_quality_mode})")
    plt.axis('off')
    
    plt.suptitle(f"Quality Adjustment Comparison (Client {client_id})\nLabels: {labels}")
    plt.tight_layout()
    
    os.makedirs('./setup/visualization', exist_ok=True)
    plt.savefig(f'./setup/visualization/quality_comparison_{args.data_quality_mode}.png', bbox_inches='tight')
    plt.close()


def visualize_client_data(dataset, client_indices, client_id, args, samples_per_client=5, title_suffix=""):
    """
    可视化客户端数据样本
    :param dataset: 数据集或Subset
    :param client_indices: 客户端的数据索引
    :param client_id: 客户端ID
    :param args: 命令行参数
    :param samples_per_client: 每个客户端显示的样本数
    :param title_suffix: 标题后缀
    """
    # 处理Subset情况
    if isinstance(dataset, torch.utils.data.Subset):
        indices = list(range(len(dataset)))  # Subset已经包含了客户端特定的索引
        samples = [dataset[i][0] for i in random.sample(indices, min(samples_per_client, len(indices)))]
        labels = [dataset[i][1] for i in random.sample(indices, min(samples_per_client, len(indices)))]
    else:
        # 原始数据集处理逻辑
        if not isinstance(client_indices, (list, tuple, set)):
            client_indices = list(client_indices)
        
        sample_indices = random.sample(client_indices, min(samples_per_client, len(client_indices)))
        samples = [dataset[int(i)][0] for i in sample_indices]
        labels = [dataset[int(i)][1] for i in sample_indices]
    
    # 转换为tensor
    if not isinstance(samples[0], torch.Tensor):
        samples = [torch.from_numpy(np.array(img)) if isinstance(img, np.ndarray) else torch.tensor(img) for img in samples]
    
    samples = torch.stack(samples)
    
    # 创建图像网格
    img_grid = make_grid(samples, nrow=samples_per_client, normalize=True)
    
    # 显示图像
    plt.figure(figsize=(10, 2))
    plt.imshow(img_grid.permute(1, 2, 0).numpy())
    title = f"Client {client_id} Samples {title_suffix}\nLabels: {labels}"
    plt.title(title)
    plt.axis('off')
    
    os.makedirs('./setup/visualization', exist_ok=True)
    plt.savefig(f'./setup/visualization/client_{client_id}_samples_{args.data_quality_mode}.png', bbox_inches='tight')
    plt.close()

# def visualize_data_distribution(dict_users, dataset, args):
#     """
#     可视化数据分布情况
#     :param dict_users: 用户数据索引字典
#     :param dataset: 完整数据集
#     :param args: 命令行参数
#     """
#     # 统计每个客户端的样本数和标签分布
#     client_stats = []
#     for client_id, indices in dict_users.items():
#         # Convert indices to plain Python integers if they're tensors/arrays
#         if isinstance(indices, (torch.Tensor, np.ndarray)):
#             indices = indices.tolist()
#         elif not isinstance(indices, (list, tuple)):
#             indices = list(indices)

#         labels = [dataset[int(i)][1] for i in indices]
#         unique, counts = np.unique(labels, return_counts=True)
#         client_stats.append({
#             'client_id': client_id,
#             'sample_count': len(indices),
#             'label_dist': dict(zip(unique, counts))
#         })
    
#     # 绘制样本数量分布
#     plt.figure(figsize=(12, 5))
    
#     # 子图1: 样本数量
#     plt.subplot(1, 2, 1)
#     sample_counts = [stat['sample_count'] for stat in client_stats]
#     plt.bar(range(args.num_users), sample_counts)
#     plt.title('Sample Count per Client')
#     plt.xlabel('Client ID')
#     plt.ylabel('Number of Samples')
    
#     # 子图2: 标签分布示例(显示前5个客户端)
#     plt.subplot(1, 2, 2)
#     for stat in client_stats[:10]:
#         plt.bar(stat['label_dist'].keys(), stat['label_dist'].values(), alpha=0.5, label=f'Client {stat["client_id"]}')
#     plt.title('Label Distribution (First 5 Clients)')
#     plt.xlabel('Class Label')
#     plt.ylabel('Count')
#     plt.legend()
    
#     plt.suptitle(f'Data Distribution (Quality: {args.data_quality_mode})')
#     plt.tight_layout()
    
#     # 保存图像
#     os.makedirs('./setup/visualization', exist_ok=True)
#     plt.savefig('./setup/visualization/data_distribution.png', bbox_inches='tight')
#     plt.close()

# def plot_transformed_samples(original, transformed, title):
#     """
#     对比显示原始和转换后的样本
#     """
#     plt.figure(figsize=(10, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.imshow(original.permute(1, 2, 0) if original.dim() == 3 else original, cmap='gray')
#     plt.title('Original')
#     plt.axis('off')
    
#     plt.subplot(1, 2, 2)
#     plt.imshow(transformed.permute(1, 2, 0) if transformed.dim() == 3 else transformed, cmap='gray')
#     plt.title('Transformed')
#     plt.axis('off')
    
#     plt.suptitle(title)
#     plt.savefig(f'./setup/visualization/{title}.png', bbox_inches='tight')
#     plt.close()