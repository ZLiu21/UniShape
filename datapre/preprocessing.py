import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from aeon.datasets import load_from_ts_file


def load_30newdata_raw_spilt(dataroot, dataset):
    train_path = os.path.join(dataroot, dataset, dataset + '_TRAIN.ts')
    test_path = os.path.join(dataroot, dataset, dataset + '_TEST.ts')

    train_x, train_target = load_from_ts_file(train_path)
    test_x, test_target = load_from_ts_file(test_path)
    num_classes = len(np.unique(np.concatenate([train_target, test_target])))

    return train_x, train_target, test_x, test_target, num_classes


def load_data_raw_spilt(dataroot, dataset):
    train = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TRAIN.tsv'), sep='\t', header=None)
    train_x = train.iloc[:, 1:]
    train_target = train.iloc[:, 0]

    test = pd.read_csv(os.path.join(dataroot, dataset, dataset + '_TEST.tsv'), sep='\t', header=None)
    test_x = test.iloc[:, 1:]
    test_target = test.iloc[:, 0]

    train_x = train_x.to_numpy(dtype=np.float32)
    train_target = train_target.to_numpy(dtype=np.float32)

    test_x = test_x.to_numpy(dtype=np.float32)
    test_target = test_target.to_numpy(dtype=np.float32)

    num_classes = len(np.unique(train_target))
    num_classes = max(num_classes, len(np.unique(test_target)))

    return train_x, train_target, test_x, test_target, num_classes


def transfer_labels(labels):
    indicies = np.unique(labels)
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indicies)[0][0]
        labels[i] = new_label

    return labels


def fill_nan_value(train_set, val_set, test_set):
    ind = np.where(np.isnan(train_set))
    col_mean = np.nanmean(train_set, axis=0)
    col_mean[np.isnan(col_mean)] = 1e-6

    train_set[ind] = np.take(col_mean, ind[1])

    ind_val = np.where(np.isnan(val_set))
    val_set[ind_val] = np.take(col_mean, ind_val[1])

    ind_test = np.where(np.isnan(test_set))
    test_set[ind_test] = np.take(col_mean, ind_test[1])
    return train_set, val_set, test_set


if __name__ == '__main__':
    pass
