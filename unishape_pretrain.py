import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import time
import torch
import numpy as np
from timm.scheduler import CosineLRScheduler
from utils.util import *
from datapre.dataloader import *
from models.pretrain_mocov3 import MoCoV3ModelPseudo
from models.unishapemodel_pretrain import UniShapeModel


os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'


def build_label_mask(targets, ratio=0.10, seed=42):
    np.random.seed(seed)
    num_samples = len(targets)
    unique_classes = np.unique(targets)
    mask = np.zeros(num_samples, dtype=bool)

    for cls in unique_classes:
        cls_indices = np.where(targets == cls)[0]
        np.random.shuffle(cls_indices)
        num_labeled = max(1, int(len(cls_indices) * ratio))  ### Dealing with imbalanced class issues
        selected = cls_indices[:num_labeled]
        mask[selected] = True

    return torch.tensor(mask)


def train_batch(model_d, epoch, idx, batch_data, batch_mask):
    samples, targets = batch_data
    sample1, sample2 = samples
    
    B = sample1.shape[0]  # batch size
    if torch.cuda.is_available():
        sample1 = sample1.cuda()
        sample2 = sample2.cuda()
        targets = targets.cuda()
        batch_mask = batch_mask.cuda()
        
    acc1, acc5 = 0.0, 0.0
    ep = epoch + idx / factor
    m = 1. - 0.5 * (1. + math.cos(math.pi * ep / 100)) * (1. - 0.999)
    loss, logits1, targets1, logits2, targets2 = model_d(sample1, sample2, targets, 0.999, train_epoch=epoch, batch_mask=batch_mask)
    
    return B, loss, acc1, acc5, model_d, logits1, targets1


def train_epoch(model_d, epoch, dataloader_dict, optimizer, scheduler, label_mask):
    iter_time, data_time = AverageMeter(), AverageMeter()
    Losses, Acc1s, Acc5s = AverageMeter(), AverageMeter(), AverageMeter()
    model_d.train()

    start_time = time.time()
    for idx, batch_data in enumerate(dataloader_dict["train"]):
        torch.cuda.synchronize()
        data_time.update(time.time() - start_time)
        
        _, _targets = batch_data
        _batch_size = _targets.shape[0]  
        
        global_idx = idx * _batch_size + torch.arange(len(_targets))  
        batch_mask = label_mask[global_idx]
        
        # train
        B, loss, acc1, acc5, model_d, fea_embds, _ = train_batch(model_d, epoch, idx, batch_data, batch_mask)
        Acc1s.update(acc1, B)
        Acc5s.update(acc5, B)
        
        loss = loss.mean()
        Losses.update(loss.item(), B)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
     
        scheduler.step_update(epoch * factor + idx)
        # measure time
        torch.cuda.synchronize()
        iter_time.update(time.time() - start_time)
        start_time = time.time()
          
        print("epoch : {}, idx: {}, step_loss: {}, fea_embds: {}".format(epoch, idx, loss, fea_embds.shape))
        
    return model_d


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed') 
    parser.add_argument('--label_ratio', type=float, default=0.1, help='') 
    parser.add_argument('--num_classes', type=int, default=1402, help='number of class') 

    parser.add_argument('--series_size', type=int, default=128)
    parser.add_argument('--window_emb_dim', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--patch_stride', type=int, default=16, help='the patch stride')

    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--epoch', type=int, default=30, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    parser.add_argument('--task', type=str, default='ssl')
    parser.add_argument('--use_eval', type=bool, default=False)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--transform_size_mode', type=str, default='fixed') ## fixed auto
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--transform_size', type=int, default=512)
    parser.add_argument('--transform_type', type=str, default='rrcrop')
    parser.add_argument('--scale', type=list, default=[0.8, 1.0]) 
    parser.add_argument('--mask_scale', type=list, default=[0.0, 0.0])
    parser.add_argument('--interpolate_mode', type=str, default='linear')  
    parser.add_argument('--mask_mode', type=str, default='sequence') 
    parser.add_argument('--window_mask_generator', type=str, default='random')
    
    parser.add_argument('--dataset_dir', type=str, default='/home/lzhen/ucr_uea_tfc_uni_dataset', help='path of UCR folder')
  
    parser.add_argument('--archives', type=str, default='ucr') 
    parser.add_argument('--train_inters', type=int, default=923) # 1845 (batch_size=1024)
    parser.add_argument('--in_channels', type=int, default=128)
    parser.add_argument('--out_channels', type=int, default=10)
    parser.add_argument('--model_series_size', type=int, default=512)
    parser.add_argument('--sample_ratio', type=float, default=1.0) 
    parser.add_argument('--norm', type=str, default='identity')
    parser.add_argument('--use_weighted_sampler', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=2048) ## 1024
    parser.add_argument('--eval_batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4)  
    parser.add_argument('--cos_moco_m', type=bool, default=True) 
    parser.add_argument('--moco_momentum', type=float, default=0.999) 
    parser.add_argument('--k', type=int, default=10)  
    
    parser.add_argument('--tau', type=float, default=0.1)  
    parser.add_argument('--proto_lamb', type=float, default=0.01) 
    parser.add_argument('--shape_ratio', type=float, default=0.6) 
    
    parser.add_argument('--scale_len', type=int, default=5)

    args = parser.parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)
    
    args.output_pretrain_dir = "./pretrained_model_ckpt/ucrueatfc_01labeled"
    
    os.makedirs(args.output_pretrain_dir, exist_ok=True)
    
    print("Output pretrain_dir = ", args.output_pretrain_dir)
    
    args.dataset = 'all-merged'
    dataloader_dict = get_dataloader(config=args)
    
    model = UniShapeModel(config=args, series_size=args.model_series_size, in_channels=args.in_channels, window_emb_dim=args.window_emb_dim, out_channels=args.out_channels, window_size=args.patch_size, stride=args.patch_stride, pre_training=True, shape_ratio=args.shape_ratio, scale_len=args.scale_len)
    model = MoCoV3ModelPseudo(backbone=model, tau=args.tau, lamb=args.proto_lamb, num_classes=args.num_classes, shape_ratio=args.shape_ratio)
    model = torch.nn.DataParallel(model).cuda()  
    
    optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
    print('Start pretraining:')

    factor = args.train_inters
    scheduler = CosineLRScheduler(optimizer, t_initial=(args.epoch - 10)*factor, lr_min=2e-05,
            warmup_t=10*factor, warmup_lr_init=2e-06, warmup_prefix=True, t_in_epochs=False)
        
    full_targets = []
    for batch_data in dataloader_dict["train"]:
        _, targets = batch_data
        full_targets.append(targets.cpu().numpy())  
    targets_np = np.concatenate(full_targets)
    label_mask = build_label_mask(targets_np, ratio=args.label_ratio, seed=args.random_seed)

    for epoch in range(1, args.epoch+1):
        model = train_epoch(model_d=model, epoch=epoch, dataloader_dict=dataloader_dict, 
                    optimizer=optimizer, scheduler=scheduler, label_mask=label_mask)
        
        if epoch % 5 == 0:
            print("epoch : {}, end ".format(epoch))
         
            model_without_ddp = model.module if hasattr(model, 'module') else model
            save_dict = {
                'epoch': epoch,
                'state_dict': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            if not os.path.exists(args.output_pretrain_dir):
                    os.makedirs(args.output_pretrain_dir)
            
            torch.save(save_dict, os.path.join(args.output_pretrain_dir, f'checkpoint_{epoch}.pth'))

    print('Finish pre-training!')
