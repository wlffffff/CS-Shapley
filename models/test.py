#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
import numpy as np
import torch.utils.data
import torch.utils.data.sampler


def test_img(net_g, datatest, args, train_test):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    if train_test == "train_dataset":
        data_loader = DataLoader(datatest, batch_size=args.bs_train)
    if train_test == "test_dataset":
        indices = list(range(len(datatest)))
        split = int(np.floor(args.test_dataset_ratio * len(datatest)))
        np.random.shuffle(indices)
        sample_test_dataset = indices[:split]
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_test_dataset)
        data_loader = DataLoader(dataset=datatest, sampler=test_sampler, batch_size=args.bs_test)
    # id = 0
    for idx, (data, target) in enumerate(data_loader):
        # id += 1
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
    # print(id)

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return float(accuracy/100), test_loss

# def test_img_everyone(net_g, datatest, idxs, args):
#     net_g.eval()
#     # testing
#     test_loss = 0
#     correct = 0
#     data_loader = DataLoader(DatasetSplit(datatest, idxs), batch_size=args.bs_test, shuffle=True)
#     l = len(data_loader)
#     for idx, (data, target) in enumerate(data_loader):
#         if args.gpu != -1:
#             data, target = data.cuda(), target.cuda()
#         log_probs = net_g(data)
#         # sum up batch loss
#         test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
#         # get the index of the max log-probability
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

#     test_loss /= len(data_loader.dataset)
#     accuracy = 100.00 * correct / len(data_loader.dataset)
#     if args.verbose:
#         print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
#             test_loss, correct, len(data_loader.dataset), accuracy))
#     return accuracy, test_loss


# def test_everyone(net_glob, dataset_train, dataset_test, dict_users_train, dict_users_test, args):
#     for idx in range(args.num_users):
#         acc_train, loss_train = test_img_everyone(net_glob, dataset_train, idxs=dict_users_train[idx], args=args)
#         acc_test, loss_test = test_img_everyone(net_glob, dataset_test, idxs=dict_users_test[idx], args=args)
#         # print(f"Client:{idx}")
#         # print("Training accuracy: {:.2f}".format(acc_train))
#         # print("Testing accuracy: {:.2f}".format(acc_test))
#         with open('./save/Result.txt', 'a') as f:
#             f.write(str(idx) + ': Training accuracy: ' + str(float(acc_train)) + ' Testing accuracy: ' + str(float(acc_test)) + '\n')
