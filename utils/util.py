import os
import sys
import random
import json
import logging

import csv
import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import classification_report
import torch
import torch.distributed as dist
import random
import torch.utils.data as data
from torch import optim, nn
from timm.optim import Lars, Lamb
from timm.scheduler import StepLRScheduler, PlateauLRScheduler, CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from sklearn.metrics import accuracy_score
from datapre.preprocessing import transfer_labels, load_data_raw_spilt, load_30newdata_raw_spilt


def set_seed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    

def cross_entropy():
    loss = nn.CrossEntropyLoss()
    return loss


def build_loss(args):
    if args.loss == 'cross_entropy':
        return cross_entropy()
    
    
def build_dataset_raw_split(args):
    train_x, train_target, test_x, test_target, num_classes = load_data_raw_spilt(args.dataroot, args.dataset)

    train_target = transfer_labels(train_target)
    test_target= transfer_labels(test_target)
    return train_x, train_target, test_x, test_target, num_classes


def build_30new_dataset_raw_split(args):
    train_x, train_target, test_x, test_target, num_classes = load_30newdata_raw_spilt(args.dataroot, args.dataset)

    train_target = transfer_labels(train_target)
    test_target= transfer_labels(test_target)
    return train_x, train_target, test_x, test_target, num_classes


def load_pretrained_model(args, model):
    checkpoint = torch.load(args.load_checkpoint_path, map_location=f'cuda:0')
    start_epoch = checkpoint['epoch'] + 1
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        if k.startswith('backbone.') and not k.startswith('backbone.fc.'):
            state_dict[k[len('backbone.'):]] = state_dict[k]
        if k.startswith('backbone.') or k.startswith('fc.'):
            del state_dict[k]

    if 'transformer_enc.pos_encoder.pe' in state_dict:
        del state_dict['transformer_enc.pos_encoder.pe']

    model_state_dict = model.state_dict()
    filtered_state_dict = {}

    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape == model_state_dict[k].shape:
                filtered_state_dict[k] = state_dict[k]
                
    msg = model.load_state_dict(filtered_state_dict, strict=False)
    return model


def getInteSet(_data_trainset):
    orig_size = _data_trainset.shape[-1] 
    new_size = 512  
    x_old = np.linspace(0, 1, orig_size) 
    x_new = np.linspace(0, 1, new_size) 
    _data_trainset = np.array([[interp1d(x_old, sample, kind='linear', axis=-1)(x_new)] for sample in _data_trainset])
    return _data_trainset


def evaluate_model(val_loader, models, num_classes):
    val_loss = 0
    val_accu = 0
    val_pred_labels = []
    val_pred_probs = []
    real_labels = []

    sum_len = 0
    for data, target in val_loader:
        with torch.no_grad():
            val_pred = torch.concat([
                torch.nn.functional.softmax(model(data, target)[0], dim=-1).unsqueeze(-1)
                for model in models
            ], dim=-1).mean(-1)
            
            for _i in  range(len(models)):
                val_loss = val_loss + models[_i](data, target)[1]
                sum_len = sum_len + len(target)
            
            val_pred_labels.append(torch.argmax(val_pred.data, axis=1).cpu().numpy())
            val_pred_probs.append(val_pred.cpu().numpy())
            real_labels.append(target.cpu().numpy())

    val_pred_labels = np.concatenate(val_pred_labels)
    real_labels = np.concatenate(real_labels)
    val_pred_probs = np.concatenate(val_pred_probs)

    return val_loss / sum_len, accuracy_score(real_labels, val_pred_labels)


class UCRDataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset.permute(0, 2, 1)  # (num_size, num_dimensions, series_length)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)
    
    
class New30Dataset(data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset.permute(0, 2, 1)  # (num_size, num_dimensions, series_length)
        self.target = target

    def __getitem__(self, index):
        return self.dataset[index], self.target[index]

    def __len__(self):
        return len(self.target)

    
class EarlyStopping:
    def __init__(self, patience=16, delta=0, best_score=None):
        self.patience = patience
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.score_max = -np.Inf
        self.delta = delta

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def dump_json_to(obj, fpath, indent=2, ensure_ascii=False, **kwconfig):
    """The helper for dumping json into the given file path"""
    with open(fpath, 'w') as fout:
        json.dump(obj, fout, indent=indent, ensure_ascii=ensure_ascii, **kwconfig)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def synchronize_between_processes(self):
        """Warning: does not synchronize the current value (i.e., self.val)!"""
        if not is_dist_avail_and_initialized(): return

        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = t[0]
        self.count = t[1]
        self.avg = self.sum / self.count if self.count > 0 else 0


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    return True if dist.is_available() and dist.is_initialized() else False


@torch.no_grad()
def concat_all_gather(x: torch.Tensor):
    """Performs all_gather operation on the provided tensors. Warning: torch.distributed.all_gather has no gradient."""
    world_size = dist.get_world_size() if is_dist_avail_and_initialized() else 1
    if world_size > 1:
        tensors_gather = [torch.ones_like(x) for _ in range(world_size)]
        dist.all_gather(tensors_gather, x, async_op=False)
        x = torch.cat(tensors_gather, dim=0)

    return x


def save_config(config):
    """Save arguments to a file.

    Parameters:
        config (argparse.Namespace): arguments

    Returns:
        str: saved arguments in text format
    """
    save_path = os.path.join(config.output_dir, 'config.txt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dump_json_to(vars(config), save_path)
    print("save config path:", save_path)


def save_model_architecture(config, model: torch.nn.Module):
    """Save model architecture to a file.

    Parameters:
        config (argparse.Namespace): arguments
        model (nn.Module): model

    Returns:
        str: saved model architecture in text format
    """
    num_params = sum(p.numel() for p in model.parameters())
    save_path = os.path.join(config.output_dir, 'architecture.txt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("save architecture path:", save_path)

    message = str(model) + f'\nNumber of parameters: {num_params}\n'
    with open(save_path, 'w') as f:
        f.write(message)

    return message


def get_scheduler(config, optimizer):
    """Return a scheduler.

    Parameters:
        config (argparse.Namespace): arguments
        optimizer (torch.optim.Optimizer): optimizer

    Returns:
        torch.optim.lr_scheduler: a scheduler
    """
    if config.scheduler == 'step':
        scheduler = StepLRScheduler(
            optimizer, decay_t=config.decay_epochs, decay_rate=config.lr_decay_factor,
            warmup_t=config.warmup_epochs, warmup_lr_init=config.warmup_lr
        )
    elif config.scheduler == 'plateau':
        scheduler = PlateauLRScheduler(
            optimizer, decay_rate=config.lr_decay_factor, patience_t=config.patience_epochs,
            verbose=False, threshold=1e-4, cooldown_t=0, mode='max',
            warmup_t=config.warmup_epochs, warmup_lr_init=config.warmup_lr, lr_min=config.min_lr
        )
    elif config.scheduler == 'cosine':
        factor = 1. if config.t_in_epochs else config.iters_per_epoch
        
        print(", factor = ", factor)
        scheduler = CosineLRScheduler(
            optimizer, t_initial=(config.num_epochs - config.warmup_epochs)*factor, lr_min=config.min_lr,
            warmup_t=config.warmup_epochs*factor, warmup_lr_init=config.warmup_lr, warmup_prefix=True,
            t_in_epochs=config.t_in_epochs
        )
    else:
        raise NotImplementedError(f'Scheduler `{config.scheduler}` is not found!')

    return scheduler