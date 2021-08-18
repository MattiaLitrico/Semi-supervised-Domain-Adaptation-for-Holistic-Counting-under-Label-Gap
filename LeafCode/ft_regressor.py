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
    parser.add_argument('--weights_fe', type=str, default='', help='Checkpoints dir') 
    parser.add_argument('--weights_cp', type=str, default='', help='Checkpoints dir')  
    parser.add_argument('--batch', type=int, default=32, help='Batch size')  
    parser.add_argument('--epochs', type=int, default=50, help='Epochs') 
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--workers', type=int, default=2, help='Workers')
    parser.add_argument('--expname', type=str, default='exp', help='Experiment name')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--train_perc', type=int, default=1, help='Training percentage')

    
    opt = parser.parse_args()

    model = CountingNetwork(output_size=1)
    
    norm_label = False
    
    if opt.data == 'Komatsuna':
        data_train = Komatsuna(opt.data, mode='train', norm_label=False, train_perc=opt.train_perc)
        data_test = Komatsuna(opt.data, mode='validation', norm_label=False, train_perc=opt.train_perc)
        data_train_loader = DataLoader(data_train, batch_size=opt.batch, num_workers=opt.workers, shuffle=True)
        data_test_loader= DataLoader(data_test, batch_size=opt.batch, num_workers=opt.workers)
    else:
        data_train = MultiModal(opt.data, mode='train', norm_label=False, train_perc=opt.train_perc)
        data_test = MultiModal(opt.data, mode='validation', norm_label=False, train_perc=opt.train_perc)
        data_train_loader = DataLoader(data_train, batch_size=opt.batch, num_workers=opt.workers, shuffle=True)
        data_test_loader= DataLoader(data_test, batch_size=opt.batch, num_workers=opt.workers)
    
    print("TRAIN STARTED")
    criterion = nn.MSELoss()
    accuracy = AbsDic()    

    e_done = 0
    best = None
    
    checkpoint_fe = load_checkpoint(opt.weights_fe)    
    checkpoint_cp = load_checkpoint(opt.weights_cp)
    load_weights(model.backbone, checkpoint_fe)
    load_weights(model.counting_part, checkpoint_cp)
    
    optimizer = SGD(model.counting_part.parameters(), opt.lr, weight_decay=0.0) 
    
    if opt.resume:
        print("Resume model from checkpoint: ", basename("logs/"+opt.expname+"/"+opt.expname+"_best.tar"))

        checkpoint_regressor = load_checkpoint("logs/"+opt.expname+"/"+opt.expname+".tar")
        
        e_done = load_epochs_done(checkpoint_regressor)
        best = load_best_value(checkpoint_regressor)
        #optimizer = load_optimizer(optimizer, checkpoint_fe)

        model.counting_part = load_weights(model.counting_part, checkpoint_regressor)
        
    train_regressor(model, criterion, accuracy, optimizer, data_train_loader, data_test_loader, opt.expname, epochs=opt.epochs,  logdir=opt.logs, e_done=e_done, best=best)
    
if __name__ == '__main__':
    main()
