import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils.util import set_seed, build_30new_dataset_raw_split, load_pretrained_model, New30Dataset, getInteSet
from sklearn.preprocessing import LabelEncoder
from models.unishapemodel_zeroshot import UniShapeModel
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


def zero_feature_extraction(val_loader, models):
    val_pred_embeds = []
    real_labels = []

    for data, target in val_loader:
        with torch.no_grad():
            val_pred = models[0](data, target)
            val_pred_embeds.append(val_pred.cpu().numpy())
            real_labels.append(target.cpu().numpy())

    val_pred_embeds = np.concatenate(val_pred_embeds)
    real_labels = np.concatenate(real_labels)
    return val_pred_embeds, real_labels


uni_30new_ts_sets = {"Tools", "SharePriceIncrease", "ShakeGestureWiimoteZ_eq", "PLAID_eq", "PickupGestureWiimoteZ_eq", "PhoneHeartbeatSound", "MelbournePedestrian_nmv", "KeplerLightCurves",
                  "GesturePebbleZ2_eq", "GesturePebbleZ1_eq", "GestureMidAirD3_eq", "GestureMidAirD2_eq", "GestureMidAirD1_eq", "FloodModeling3_disc", "FloodModeling2_disc", "FloodModeling1_disc",
                  "ElectricDeviceDetection", "DodgerLoopWeekend_nmv", "DodgerLoopGame_nmv", "DodgerLoopDay_nmv", "Covid3Month_disc", "Colposcopy", "AsphaltRegularityUni_eq", "AsphaltPavementTypeUni_eq",
                  "AsphaltObstaclesUni_eq", "AllGestureWiimoteZ_eq", "AllGestureWiimoteY_eq", "AllGestureWiimoteX_eq", "AconityMINIPrinterSmall_eq", "AconityMINIPrinterLarge_eq"}

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # Base setup
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup  
    parser.add_argument('--dataset', type=str, default="Tools", help='dataset(in ucr)')  # "Tools"
    parser.add_argument('--dataroot', type=str, default='./30NewDatasets', help='path of UCR folder')
    parser.add_argument('--num_class', type=int, default=0, help='number of class')
    
    # Model
    parser.add_argument('--in_channels', type=int, default=128)
    parser.add_argument('--window_emb_dim', type=int, default=128)
    parser.add_argument('--window_size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--scale_len', type=int, default=4)

    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=512, help='')
    parser.add_argument('--epoch', type=int, default=300, help='training epoch')
    parser.add_argument('--cuda', type=str, default='cuda:0')

    args = parser.parse_args()

    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    set_seed(args)

    train_x, train_target, test_x, test_target, num_classes = build_30new_dataset_raw_split(args)
    train_x = train_x.transpose(0, 2, 1)
    test_x = test_x.transpose(0, 2, 1)
  
    args.num_class = num_classes
    args.seq_len = train_x.shape[1]
    args.model_series_size = train_x.shape[1]
    args.input_size = train_x.shape[2]
    args.model_series_size = 512
    args.seq_len = 512
    args.out_channels = num_classes
    args.load_checkpoint_path = './pretrained_model_ckpt/unishape_checkpoint_zeroshot.pth'
    
    model_list = []
    
    for _ in range(1):  ## Only one model
    
        model = UniShapeModel( config=args,
            series_size=args.model_series_size, in_channels=args.in_channels, window_emb_dim=args.window_emb_dim,
            out_channels=args.out_channels, window_size=args.window_size, stride=args.stride, shape_alpha=0.01, shape_sparse_ratio=0.6,
            scale_len=args.scale_len
        )
        
        model = load_pretrained_model(args, model)
        model = model.to(device)
        model_list.append(model)
  
    print('Start zero shot feature extraction and evaluate: ')
   
    train_dataset = getInteSet(train_x.transpose(0, 2, 1))
    train_dataset =  np.squeeze(train_dataset, axis=2).transpose(0, 2, 1) 
    test_dataset = getInteSet(test_x.transpose(0, 2, 1))
    test_dataset =  np.squeeze(test_dataset, axis=2).transpose(0, 2, 1)
    
    le = LabelEncoder()
    train_target = le.fit_transform(train_target)
    test_target = le.transform(test_target)

    train_set = New30Dataset(torch.from_numpy(train_dataset).type(torch.FloatTensor).to(device),
                           torch.from_numpy(train_target).type(torch.FloatTensor).to(device).to(torch.int64))
    test_set = New30Dataset(torch.from_numpy(test_dataset).type(torch.FloatTensor).to(device),
                          torch.from_numpy(test_target).type(torch.FloatTensor).to(device).to(torch.int64))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0)
 
    train_embeds, train_labels = zero_feature_extraction(train_loader, model_list)
    test_embeds, test_labels = zero_feature_extraction(test_loader, model_list)
    
    predictor = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    predictor.fit(train_embeds, train_labels)
    
    y_pred = predictor.predict(test_embeds)
    test_accuracy = accuracy_score(test_labels, y_pred)
    
    print("test_accuracy = ", test_accuracy)
    print('Done!')
