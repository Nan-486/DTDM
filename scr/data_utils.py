import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os

from tf_keras.src.backend import dtype
from torch.utils.data import Dataset

def data_load(train_path, valid_path, test_path):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)
    #print(f'type:{type(train_list)}')
    # print(f"train_list.shapr{train_list.shape}")
    # print(f"valid_list.shapr{valid_list.shape}")
    # print(f"test_list.shapr{test_list.shape}")

    train_list=train_list[:train_list.shape[0]//2]
    valid_list = valid_list[:valid_list.shape[0]//2]
    test_list = test_list[:test_list.shape[0]//2]

    # print(f"train_list_2.shapr{train_list.shape}")
    # print(f"valid_list_2.shapr{valid_list.shape}")
    # print(f"test_list_2.shapr{test_list.shape}")


    uid_max_train,uid_max_test,uid_max_valid = 0,0,0
    iid_max_train,iid_max_test,iid_max_valid = 0,0,0
    train_dict,test_dict,valid_dict = {},{},{}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max_train:
            uid_max_train = uid
        if iid > iid_max_train:
            iid_max_train = iid

    for uid, iid in test_list:
        if uid not in test_dict:
            test_dict[uid] = []
        test_dict[uid].append(iid)
        if uid > uid_max_test:
            uid_max_test = uid
        if iid > iid_max_test:
            iid_max_test = iid

    for uid, iid in valid_list:
        if uid not in valid_dict:
            valid_dict[uid] = []
        valid_dict[uid].append(iid)
        if uid > uid_max_valid:
            uid_max_valid = uid
        if iid > iid_max_valid:
            iid_max_valid = iid
    
    n_user_train = uid_max_train + 1
    n_item_train = iid_max_train + 1
    print(f'user num: {n_user_train}')
    print(f'item num: {n_item_train}')

    # print(f'uid_max_valid num: {uid_max_valid}')
    # print(f'iid_max_valid num: {iid_max_valid}')


    train_data = sp.csr_matrix((np.ones_like(train_list[:, 0]), \
        (train_list[:, 0], train_list[:, 1])), dtype='float64', \
        shape=(n_user_train, iid_max_valid+1))
    
    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user_train, iid_max_valid+1))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user_train, iid_max_valid+1))  # test_groundtruth
    
    return train_data, valid_y_data, test_y_data, n_user_train, iid_max_valid+1


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)
