import os
import glob
import random
from collections import Counter

from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
import torch


def prepareUCR(dataset_dir: str, fill_missing_and_variable=True, normalize=False, grain=10000):
    """Prepare UCR sub-datasets.

    Download the original dataset archive from https://www.cs.ucr.edu/~eamonn/time_series_data_2018/ and
    extract it in the `dataset_dir` directory.

    Parameters:
        dataset_dir (str): directory of UCR archive
        fill_missing_and_variable (bool): whether to use repaired values provided by UCR-128 archive
        normalize (bool): whether to normalize sub-datasets by its mean and std when merging
        test_ratio (float): test dataset ratio
        grain (int): grain size of splitting the huge merged dataset into multiple files
    """
    assert os.path.exists(dataset_dir), f'UCR archive directory `{dataset_dir}` does not exist!'

    # delete these sub-datasets to merge the whole UCR archive without duplication
    DEL_DATASETS = [
        'Chinatown', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxOutlineCorrect', 
        'DodgerLoopGame', 'DodgerLoopWeekend', 'FreezerSmallTrain', 'GesturePebbleZ2',
        'GunPointMaleVersusFemale', 'GunPointOldVersusYoung', 'InsectEPGSmallTrain',
        'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect', 'MixedShapesSmallTrain',
        'PhalangesOutlinesCorrect', 'PickupGestureWiimoteZ', 'ProximalPhalanxOutlineAgeGroup',
        'ProximalPhalanxOutlineCorrect', 'SemgHandGenderCh2', 'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ', 'WormsTwoClass'
    ]

    root = os.path.abspath(os.path.join(dataset_dir, '..'))
    save_dir = os.path.join(root, 'ucr-128')
    merge_dir, merge_split_dir = os.path.join(root, 'ucr-merged'), os.path.join(root, 'ucr-merged-split')

    MVVL_DIR = 'Missing_value_and_variable_length_datasets_adjusted'
    MVVL_DATASETS = os.listdir(os.path.join(dataset_dir, MVVL_DIR))

    DATASETS = os.listdir(dataset_dir)
    DATASETS.remove(MVVL_DIR)
    assert len(DATASETS) == 128, f'UCR archive directory `{dataset_dir}` should contain 128 sub-datasets, but found {len(DATASETS)}!'

    all_data = pd.DataFrame()  # split data into train and test randomly
    train_split, test_split = pd.DataFrame(), pd.DataFrame()  # split data into train and test by the original dataset

    for d in tqdm(sorted(DATASETS)):
        if d in MVVL_DATASETS and fill_missing_and_variable:
            # missing values are treated with linear interpolation
            # variable length are treated with low-amplitude noise padding
            df_train = pd.read_csv(os.path.join(dataset_dir, MVVL_DIR, d, f'{d}_TRAIN.tsv'), sep='\t', header=None)
            df_test = pd.read_csv(os.path.join(dataset_dir, MVVL_DIR, d, f'{d}_TEST.tsv'), sep='\t', header=None)
        else:
            df_train = pd.read_csv(os.path.join(dataset_dir, d, f'{d}_TRAIN.tsv'), sep='\t', header=None)
            df_test = pd.read_csv(os.path.join(dataset_dir, d, f'{d}_TEST.tsv'), sep='\t', header=None)

        # prepare data in sub-datasets, i.e., change category names and file suffix (.tsv -> .csv)
        data = pd.concat([df_train, df_test], axis=0)
        if d in ['FordA', 'FordB']:  # merge categories in the two datasets
            new_idx = {sorted(data.iloc[:, 0].unique())[i]: f'Ford_{i:02d}' for i in range(len(data.iloc[:, 0].unique()))}
        else:
            new_idx = {sorted(data.iloc[:, 0].unique())[i]: f'{d}_{i:02d}' for i in range(len(data.iloc[:, 0].unique()))}
        data.iloc[:, 0] = data.iloc[:, 0].map(new_idx)
        df_train.iloc[:, 0] = df_train.iloc[:, 0].map(new_idx)
        df_test.iloc[:, 0] = df_test.iloc[:, 0].map(new_idx)

        os.makedirs(os.path.join(save_dir, d), exist_ok=True)
        df_train.columns, df_test.columns = [str(i) for i in range(df_train.shape[1])], [str(i) for i in range(df_test.shape[1])]
        df_train.reset_index(drop=True).to_feather(os.path.join(save_dir, d, f'train.feather'))
        df_test.reset_index(drop=True).to_feather(os.path.join(save_dir, d, f'test.feather'))

        # merge sub-datasets
        # if d not in DEL_DATASETS:
        #     if normalize:  # normalize each sub-dataset by its mean and std of the whole dataset
        #         data_mean, data_std = data.iloc[:, 1:].values.mean(), data.iloc[:, 1:].values.std()
        #         data.iloc[:, 1:] = (data.iloc[:, 1:] - data_mean) / (data_std + 1e-8)
        #     all_data = pd.concat([all_data, data], axis=0, sort=True)
        #     if normalize:  # normalize each sub-dataset by its mean and std of the training dataset for the split version
        #         train_mean, train_std = df_train.iloc[:, 1:].values.mean(), df_train.iloc[:, 1:].values.std()
        #         df_train.iloc[:, 1:] = (df_train.iloc[:, 1:] - train_mean) / (train_std + 1e-8)
        #         df_test.iloc[:, 1:] = (df_test.iloc[:, 1:] - train_mean) / (train_std + 1e-8)
        #     train_split, test_split = pd.concat([train_split, df_train], axis=0, sort=True), pd.concat([test_split, df_test], axis=0, sort=True)
        
    # save uni-dimensional data as `.feather` file
    # os.makedirs(os.path.join(merge_dir), exist_ok=True)
    # all_samples.columns = [str(i) for i in range(all_samples.shape[1])]
    # all_samples.sample(frac=1, random_state=42)  # shuffle the data
    # for i in tqdm(range(all_samples.shape[0]//grain + 1)):
    #     end = min((i+1)*grain, all_samples.shape[0])
    #     all_samples.iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(merge_dir, f'train_{i}.feather'))
    # all_samples.reset_index(drop=True).to_feather(os.path.join(merge_dir, 'train.feather'))


def prepareUEA(dataset_dir: str, normalize=False, grain=10000):
    """Prepare UEA sub-datasets.

    Download the original dataset archive from http://www.timeseriesclassification.com/dataset.php and
    extract it in the `dataset_dir` directory.

    Parameters:
        dataset_dir (str): directory of UEA archive
        normalize (bool): whether to normalize sub-datasets by its mean and std when merging
        grain (int): grain size of splitting the huge merged dataset into multiple files
    """
    assert os.path.exists(dataset_dir), f'UEA archive directory `{dataset_dir}` does not exist!'

    DATASETS = os.listdir(dataset_dir)
    DATASETS = list(set(DATASETS) - set(['Descriptions', 'Images', 'DataDimensions.csv', 'UEAArchive2018.pdf']))  # remove unnecessary files
    assert len(DATASETS) == 30, f'UEA archive directory `{dataset_dir}` should contain 30 sub-datasets, but found {len(DATASETS)}!'

    root = os.path.abspath(os.path.join(dataset_dir, '..'))
    save_dir, merge_dir = os.path.join(root, 'uea-30'), os.path.join(root, 'uea-merged')

    all_samples = pd.DataFrame()
    for d in tqdm(sorted(DATASETS)):
        train = loadarff(os.path.join(dataset_dir, d, f'{d}_TRAIN.arff'))[0]
        samples, labels = [], []
        for sample, label in train:  # instance-loop
            sample = np.array([s.tolist() for s in sample])  # (d, l)
            label = label.decode("utf-8")  # (1,)
            samples.append(sample)
            labels.append(label)
        samples = torch.tensor(samples, dtype=torch.float)  # (n, d, l)
        classes = set(labels)  # unique labels
        class_to_idx = {c: i for i, c in enumerate(classes)}  # map label (string) to index (int)
        labels = torch.tensor([class_to_idx[l] for l in labels], dtype=torch.long)  # (n,)
        data = {'samples': samples, 'labels': labels}

        test = loadarff(os.path.join(dataset_dir, d, f'{d}_TEST.arff'))[0]
        samples_test, labels_test = [], []
        for sample, label in test:  # instance-loop
            sample = np.array([s.tolist() for s in sample])  # (d, l)
            label = label.decode("utf-8")  # (1,)
            samples_test.append(sample)
            labels_test.append(label)
        samples_test = torch.tensor(samples_test, dtype=torch.float)  # (n, d, l)
        labels_test = torch.tensor([class_to_idx[l] for l in labels_test], dtype=torch.long)  # (n,)
        data_test = {'samples': samples_test, 'labels': labels_test}

        # save multi-dimensional data as `.pt` file
        os.makedirs(os.path.join(save_dir, d), exist_ok=True)
        torch.save(data, os.path.join(save_dir, d, f'train.pt'))
        torch.save(data_test, os.path.join(save_dir, d, f'test.pt'))

        # if d not in ['EigenWorms', 'FaceDetection', 'InsectWingbeat', 'PenDigits']:  # do not merge these datasets for some reasons
        #     if normalize:  # normalize data by training set mean and std per channel
        #         train_mean, train_std = samples.mean(dim=(0, 2)), samples.std(dim=(0, 2))
        #         if torch.isnan(train_mean).any() or torch.isnan(train_std).any():  # NaN in the dataset
        #             train_mean, train_std = np.nanmean(samples.numpy(), axis=(0, 2)), np.nanstd(samples.numpy(), axis=(0, 2))
        #         train_mean = torch.as_tensor(train_mean, dtype=torch.float).view(-1, 1)
        #         train_std = torch.as_tensor(train_std, dtype=torch.float).view(-1, 1)
        #         samples = (samples - train_mean) / (train_std + 1e-8)
        #         samples_test = (samples_test - train_mean) / (train_std + 1e-8)

        #     samples_flat = samples.reshape(-1, samples.shape[-1]).numpy()  # (n*d, l)
        #     samples_test_flat = samples_test.reshape(-1, samples_test.shape[-1]).numpy()  # (n*d, l)
        #     all_samples = pd.concat([all_samples, pd.DataFrame(samples_flat), pd.DataFrame(samples_test_flat)])

    # save uni-dimensional data as `.feather` file
    # os.makedirs(os.path.join(merge_dir), exist_ok=True)
    # all_samples.columns = [str(i) for i in range(all_samples.shape[1])]
    # all_samples.sample(frac=1, random_state=42)  # shuffle the data
    # for i in tqdm(range(all_samples.shape[0]//grain + 1)):
    #     end = min((i+1)*grain, all_samples.shape[0])
    #     all_samples.iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(merge_dir, f'train_{i}.feather'))
    # all_samples.reset_index(drop=True).to_feather(os.path.join(merge_dir, 'train.feather'))
    
    
def load_pt_dataset(data_path):
    data = torch.load(data_path)
    try:
        samples = data['samples'].float()  # torch tensor: (n_samples, n_features, length)
    except:
        samples = torch.Tensor(data['samples']).float()
    try:
        targets = data['labels'].long()  # torch tensor: (n_samples,)
    except:
        targets = torch.Tensor(data['labels']).long()  # torch tensor: (n_samples,)

    return samples, targets


def load_feather_dataset(data_path):
    data = pd.read_feather(data_path)
    classes = sorted(data.iloc[:, 0].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    data[data.columns[0]] = data[data.columns[0]].map(class_to_idx)

    samples = torch.Tensor(data.iloc[:, 1:].values).float()
    targets = torch.Tensor(data.iloc[:, 0].values).long()

    if samples.ndim == 2:
        samples.unsqueeze_(1)

    return samples, targets


def mergeDatasets(dataset_dir_dict: dict, save_dir: str, grain=10000):
    """merge provided datasets with original split of train test set.

    Parameters:
        dataset_dir_list (dict): dictionery of the directory of source sub-datasets
        save_dir (str): save directory of the merged dataset
        grain (int): grain size of splitting the huge merged dataset into multiple files
    """
    # get sub-dataset dir list 
    UEA_DIR_NAME = 'uea-30'   ### This name must match the folder name generated after executing the prepareUEA() method.
    UEA_DEL_DATASETS = ['InsectWingbeat', 'EigenWorms']
    # UEA_DEL_DATASETS = []
    MAX_LENGTH_LIMIT = 2048
    DATASETS = []
    merge_dir = os.path.join(save_dir, 'all-merged')
    os.makedirs(merge_dir, exist_ok=True)
    # os.makedirs(merge_shuffle_dir, exist_ok=True)
    for directory, dataset_type in dataset_dir_dict.items():
        if dataset_type == 'single':
            DATASETS.append(directory)
        else:
            for d in os.listdir(directory):
                if directory.split('/')[-1] == UEA_DIR_NAME and d in UEA_DEL_DATASETS:
                    continue
                DATASETS.append(os.path.join(directory, d))
    
    # load sub-datasets
    merged_df = {}
    df_data = {}
    df_no_duplicate_data = []
    for t in ['train', 'test']:
        df_data[t] = []
    for d in tqdm(sorted(DATASETS)):
        archieve_name = d.split("/")[-2].split("-")[0]
        dataset_name = d.split("/")[-1]
        uniq_name = archieve_name + "-" + dataset_name
        data_dict = {}
        df_data_dict = {}
        val_set_exist = True
        for t in ['train', 'test']:
            # load data
            try:
                try:
                    samples, targets = load_pt_dataset(os.path.join(d, f'{t}.pt'))
                except:
                    samples, targets = load_feather_dataset(os.path.join(d, f'{t}.feather'))
            except:
                val_set_exist = False
            N, C, L = samples.shape
            samples = samples.reshape(N*C, L)
            if L > MAX_LENGTH_LIMIT:
                # samples = samples[:, :L]
                samples = samples[:, :MAX_LENGTH_LIMIT]
            targets = np.repeat(targets, C, axis=0)
            data_dict[t] = {'samples': pd.DataFrame(samples.numpy()), 'targets': pd.DataFrame(targets.numpy())}
      
        # reindex classes
        ori_idx = data_dict['train']['targets'].iloc[:, 0].unique()
        new_idx = {sorted(ori_idx)[i]: f'{uniq_name}_{i:02d}' for i in range(len(ori_idx))}
        # concat target and feature
        for t in ['train', 'test']:
            data_dict[t]['targets'].iloc[:, 0] = data_dict[t]['targets'].iloc[:, 0].map(new_idx)
            df_data_dict[t] = pd.concat([data_dict[t]['targets'], data_dict[t]['samples']], axis=1)
            df_data_dict[t].columns = [str(i) for i in range(df_data_dict[t].shape[1])]
            df_data[t].append(df_data_dict[t])
        print(f'Finish process dataset:{uniq_name} '
                f'with train/test size: {df_data_dict["train"].shape[0]}/{df_data_dict["test"].shape[0]}, '
                f'features num: {C}, '
                f'class num: {len(new_idx)}, '
                f'length: {df_data_dict["train"].shape[1]}')

    # merge all datasets
    for t in ['train', 'test']:
        merged_df[t] = pd.concat(df_data[t], axis=0)    # sort=True will reorder the sequence
        merged_df[t].columns = [str(i) for i in range(merged_df[t].shape[1])]
    print(f'Saving processed data: '
            f'{len(merged_df["train"])} samples in train set, '
            # f'{len(merged_df["val"])} samples in val set, '
            # f'{len(merged_df["test"])} samples in test set, '
            f'max_length: {merged_df["train"].shape[1]}, '
            f'total_classes: {len(merged_df["train"].iloc[:, 0].unique())}')

    # save original split data
    for t in ['train', 'test']:
        # for i in tqdm(range(merged_df[t].shape[0]//grain + 1)):
        #     end = min((i+1)*grain, merged_df[t].shape[0])
        #     merged_df[t].iloc[i*grain:end].reset_index(drop=True).to_feather(os.path.join(merge_dir, f'{t}_{i}.feather'))
        print(f'Start save whole {t} merged data')
        merged_df[t].reset_index(drop=True).to_feather(os.path.join(merge_dir, f'{t}.feather'))
           

if __name__ == "__main__":
    prepareUCR(dataset_dir="/home/lzhen/UCRArchive_2018")
    prepareUEA(dataset_dir="/home/lzhen/Multivariate2018_arff")
    
    print("Finsh preprocess! Begin merge:")
    # merge all datasets 
    dataset_dir_dict = {
        '/home/lzhen/ucr-128': 'multi',  ## Saving processed data: 60561 samples in train set, total_classes: 1118; only ucr for pretrain
        '/home/lzhen/uea-30': 'multi',  ## Saving processed data: 1386874 samples in train set, total_classes: 250; only uea for pretrain
        '/home/lzhen/tfc-8': 'multi',  ### can be download by our provide link: https://drive.google.com/file/d/1J2AiL2KgDpZGprnWfH-cqza-Dih25p5z/view?usp=sharing
    }   ## Saving processed data: 1889192 samples in train set, total_classes: 1402; all (ucr + uea + tfc-8) for pretrain
   
    save_dir = '/home/lzhen/ucr_uea_tfc_uni_dataset/'
    mergeDatasets(dataset_dir_dict, save_dir, grain=100000)
    
    print("Merge Success!")