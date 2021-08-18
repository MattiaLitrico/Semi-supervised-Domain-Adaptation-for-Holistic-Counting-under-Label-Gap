from torch.utils.data import DataLoader
from models import *
from torchvision import transforms
from torch.optim import SGD, Adam
from os.path import join, basename, splitext
from metrics import *
import torch
import numpy as np
from PIL import Image
from torch import nn
import argparse
import time
from dataset import *
from network import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='Data dir')  
    parser.add_argument('--logs', type=str, default='logs', help='Logs dir') 
    parser.add_argument('--batch', type=int, default=32, help='Batch size')  
    parser.add_argument('--epochs', type=int, default=50, help='Epochs') 
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--workers', type=int, default=2, help='Workers')
    parser.add_argument('--expname', type=str, default='exp', help='Experiment name')
    parser.add_argument('--resume', action='store_true', help='resume training')
    
    opt = parser.parse_args()

    model = CountingNetwork()
    
    norm_label = False
    
    data = dict()
    for i in range(4):
        data_train = Komatsuna(opt.data, mode='train', norm_label=norm_label)
        data_test = Komatsuna(opt.data, mode='validation', norm_label=norm_label)
        data_train_loader = DataLoader(data_train, batch_size=opt.batch, num_workers=opt.workers, shuffle=True)
        data_test_loader= DataLoader(data_test, batch_size=opt.batch, num_workers=opt.workers)

        train_split_name = "train_split_" + str(i)
        test_split_name = "test_split_" + str(i)
        
        data[train_split_name] = data_train_loader
        data[test_split_name] = data_test_loader
        
    print("TRAIN STARTED")
    criterion = nn.MSELoss()
    accuracy = AbsDic()    
    optimizer = Adam(model.parameters(), opt.lr, weight_decay=0.01)
    #optimizer = SGD(model.parameters(), opt.lr, momentum=0.9)
    e_done = 0
    best = None
    
    if opt.resume:
        print("Resume model from checkpoint: ", basename("logs/"+opt.expname+"/"+opt.expname+"_fe.tar"))
        print("Resume model from checkpoint: ", basename("logs/"+opt.expname+"/"+opt.expname+"_cp.tar"))
        checkpoint_fe = load_checkpoint("logs/"+opt.expname+"/"+opt.expname+"_fe.tar")
        checkpoint_cp = load_checkpoint("logs/"+opt.expname+"/"+opt.expname+"_cp.tar")
        
        e_done = load_epochs_done(checkpoint_fe)
        best = load_best_value(checkpoint_fe)
        #optimizer = load_optimizer(optimizer, checkpoint_fe)

        model.backbone = load_weights(model.backbone, checkpoint_fe)
        model.counting_part = load_weights(model.counting_part, checkpoint_cp)

    train(model, criterion, accuracy, optimizer, data, opt.expname, epochs=opt.epochs,  logdir=opt.logs, e_done=e_done, best=best)
    
if __name__ == '__main__':
    main()
