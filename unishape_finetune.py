import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.util import set_seed, build_dataset_raw_split, load_pretrained_model, evaluate_model, UCRDataset
from datapre.preprocessing import fill_nan_value
from models.unishapemodel_finetune import UniShapeModel
from utils.hyp_batch_shape_sizes import ucr128_hyp_dict_shape_size_batch_size


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    parser.add_argument('--dataset', type=str, default='CBF', help='dataset(in ucr)') 
    parser.add_argument('--dataroot', type=str, default='/home/lzhen/UCRArchive_2018', help='path of UCR folder')
    parser.add_argument('--num_class', type=int, default=0, help='number of class')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    parser.add_argument('--in_channels', type=int, default=128)
    parser.add_argument('--window_emb_dim', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--shape_ratio', type=float, default=0.1)
    parser.add_argument('--scale_len', type=int, default=5) ## 1， 2， 3， 4， 5
    parser.add_argument('--ensemble_num_model', type=int, default=5) ## 1
    parser.add_argument('--pretrain_label_ratio', type=float, default=0.1, help='')

    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=512, help='')
    parser.add_argument('--epoch', type=int, default=300, help='training epoch') 
    parser.add_argument('--cuda', type=str, default='cuda:0')

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)
    
    train_x, train_target, test_x, test_target, num_classes = build_dataset_raw_split(args)
    args.num_class = num_classes
    if train_x.shape[0] < args.batch_size:
        args.batch_size = train_x.shape[0]

    train_x = train_x[:, :, np.newaxis]
    test_x = test_x[:, :, np.newaxis]

    args.batch_size = ucr128_hyp_dict_shape_size_batch_size[args.dataset]["batch_size"]
    args.scale_len = ucr128_hyp_dict_shape_size_batch_size[args.dataset]["scale_len"]  

    args.num_class = num_classes
    args.seq_len = train_x.shape[1]
    args.model_series_size = train_x.shape[1]
    args.input_size = train_x.shape[2]
    args.out_channels = num_classes
    
    train_dataset, val_dataset, test_dataset = fill_nan_value(train_x, train_x, test_x)
    val_target = train_target

    print("args.batch_size = ", args.batch_size, ", train_x.shape = ", train_x.shape)

    args.load_checkpoint_path = './pretrained_model_ckpt/unishape_checkpoint_finetune.pth'
    
    model_list = []
    optimizer_list = []  
    for _ in range(args.ensemble_num_model):
        model = UniShapeModel(config=args,
            series_size=args.model_series_size, in_channels=args.in_channels, window_emb_dim=args.window_emb_dim,
            out_channels=args.out_channels, window_size=args.window_size, stride=args.stride, pre_training=False, shape_alpha=0.01, shape_ratio=0.6,
            scale_len=args.scale_len
        )
        model = load_pretrained_model(args, model)
        model = model.to(device)
        optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
        model_list.append(model)
        optimizer_list.append(optimizer)

    test_accuracies = []
    print('Start finetuning: ')

    train_set = UCRDataset(torch.from_numpy(train_dataset).type(torch.FloatTensor).to(device),
                           torch.from_numpy(train_target).type(torch.FloatTensor).to(device).to(torch.int64))
    val_set = UCRDataset(torch.from_numpy(val_dataset).type(torch.FloatTensor).to(device),
                         torch.from_numpy(val_target).type(torch.FloatTensor).to(device).to(torch.int64))
    test_set = UCRDataset(torch.from_numpy(test_dataset).type(torch.FloatTensor).to(device),
                          torch.from_numpy(test_target).type(torch.FloatTensor).to(device).to(torch.int64))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)

    train_loss = []
    train_accuracy = []
    num_steps = args.epoch // args.batch_size

    last_loss = float('inf')
    stop_count = 0
    increase_count = 0
    num_steps = train_set.__len__() // args.batch_size + 1
    min_val_loss = float('inf')
    test_accuracy = 0
    
    for epoch in range(args.epoch):
        epoch_train_loss = 0
        num_iterations = 0
        
        for m_j in range(args.ensemble_num_model):
            model_list[m_j].train()
            for x, y in train_loader:
                optimizer_list[m_j].zero_grad()
                pred, step_loss = model_list[m_j](x, y)
                step_loss.backward()
                optimizer_list[m_j].step()
                epoch_train_loss += step_loss.item()
                num_iterations += 1

            epoch_train_loss /= num_steps
     
        for m_j in range(args.ensemble_num_model):
            model_list[m_j].eval()
        
        val_loss, val_accu = evaluate_model(val_loader, model_list, num_classes)
        if min_val_loss > val_loss:
            min_val_loss = val_loss
            end_val_epoch = epoch
            test_loss, test_accuracy = evaluate_model(test_loader, model_list, num_classes)

        if abs(last_loss - val_loss) <= 1e-4:
            stop_count += 1
        else:
            stop_count = 0

        if val_loss > last_loss:
            increase_count += 1
        else:
            increase_count = 0

        last_loss = val_loss

        if epoch % 50 == 0: ## 50 
            print("epoch : {}, train loss: {} , test_accuracy : {}".format(epoch, epoch_train_loss,test_accuracy))

    print("Training end: test_accuracy = ", test_accuracy)
    print('Finish finetuning!')
