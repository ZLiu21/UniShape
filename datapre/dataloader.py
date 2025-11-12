import functools

from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from collections import defaultdict
from .datasets import *
from .transforms import *
import numpy as np
import torch
import torch.utils.data as data


class UCRDataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        if len(self.dataset.shape) == 2:
            self.dataset = torch.unsqueeze(self.dataset, 1)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)


def get_dataset_cls(config):
    pt_keywords = ['uea-', 'TFC-']
    data_format = 'feather'
    for kw in pt_keywords:
        if kw in config.dataset or kw in config.dataset_dir:
            data_format = 'pt'
            break
    config.data_format = data_format
    dataset_cls = functools.partial(CustomDataset)

    return dataset_cls, data_format


def get_transform(config):
    transform_dict = {}
    if config.transform_type == 'rrcrop':
        # augmentation for train
        train_transform = TSCompose([
            TSRandomResizedCrop(size=config.transform_size, scale=config.scale, mode=config.interpolate_mode),
            TSRandomMask(scale=config.mask_scale, mask_mode=config.mask_mode,
                        win_size=config.window_size, window_mask_generator=config.window_mask_generator),
        ])

        # resize for test
        test_transform = TSCompose([
            TSResize(size=config.transform_size, mode=config.interpolate_mode),
        ])
        
    else:   # config.transform_type == 'identity'
        train_transform = TSIdentity()
        test_transform = TSIdentity()

    # get contrasive samples for self-supervised learning
    if config.task == 'ssl':
        train_transform = TSTwoViewsTransform(train_transform)
        
        print("train_transform = ", train_transform)

    transform_dict['train'] = train_transform
    transform_dict['val'] = train_transform    ## cannot use test set
    transform_dict['test'] = test_transform

    return transform_dict


def normalize_dataset(config, dataset_dict):
    eps = 1e-8
    if config.norm == 'global':
        samples = dataset_dict['train'].samples
        mean, std = np.nanmean(samples.numpy(), axis=(0, 2)), np.nanstd(samples.numpy(), axis=(0, 2))
        mean = torch.as_tensor(mean).view(-1, 1)
        std = torch.as_tensor(std).view(-1, 1)
        for type in config.dataset_type_list:
            dataset_dict[type].samples = (dataset_dict[type].samples - mean) / (std + eps)
    elif config.norm == 'instance':
        num_channels = dataset_dict['train'].num_channels
        for type in config.dataset_type_list:
            mean, std = np.nanmean(dataset_dict[type].samples.numpy(), axis=(2)), np.nanstd(dataset_dict[type].samples.numpy(), axis=(2))
            mean = torch.as_tensor(mean).view(-1, num_channels, 1)
            std = torch.as_tensor(std).view(-1, num_channels, 1)
            dataset_dict[type].samples = (dataset_dict[type].samples - mean) / (std + eps)


def get_dataset(config):
    transform_dict = get_transform(config)
    dataset_dict = {}
    for type in ['train', 'test']:
        dataset_dict[type] = CustomDataset(config=config,
                                         type=type,
                                         transform=transform_dict[type])
    try:
        config.no_validation_set = False
        dataset_dict['val'] = CustomDataset(config=config,
                                      type='val',
                                      transform=transform_dict['val'])
    except:
        config.no_validation_set = True
        dataset_dict['val'] = CustomDataset(config=config,
                                        type='train',
                                        transform=transform_dict['train'])    ## cannot use test set
        
    config.dataset_type_list = dataset_dict.keys()
    normalize_dataset(config, dataset_dict)

    return dataset_dict


def get_weighted_sampler(config, dataset):
    class_counts = torch.bincount(dataset.targets)
    sample_weights = 1 / class_counts[dataset.targets]

    return WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)


def build_dataloader(config, dataset_dict):
    dataloader_dict = {}
    for type in config.dataset_type_list:
        sampler = None
        if config.use_weighted_sampler and type == 'train':
            sampler = get_weighted_sampler(config, dataset_dict[type])
        dataloader_dict[type] = DataLoader(
            dataset=dataset_dict[type],
            shuffle=True if type == 'train' and sampler is None else False,
            batch_size=config.batch_size if type == 'train' else config.eval_batch_size,
            num_workers=config.num_workers,
            sampler = sampler
        )

    return dataloader_dict


def get_dataloader(config):
    # get dataset
    dataset_dict = get_dataset(config)
    # get dataloader
    dataloader_dict = build_dataloader(config, dataset_dict)
    # set config
    config.ori_series_size = dataset_dict['train'].series_size
    config.model_series_size = config.transform_size
    config.num_classes = dataset_dict['train'].num_classes
    config.num_channels = dataset_dict['train'].num_channels
    config.iters_per_epoch = len(dataloader_dict['train'])
    
    # train_dataset = dataset_dict['train']
    # print("Total samples in train dataset:", len(train_dataset))
    # val_dataset = dataset_dict['val']
    # print("Total samples in val dataset:", len(val_dataset))
    # test_dataset = dataset_dict['test']
    # print("Total samples in test dataset:", len(test_dataset))
    # print("config.ori_series_size = ", config.ori_series_size)
    # print("config.model_series_size = ", config.model_series_size)
    # print("config.num_classes = ", config.num_classes)
    # print("config.num_channels = ", config.num_channels)
    # print("config.iters_per_epoch = ", config.iters_per_epoch)

    return dataloader_dict
