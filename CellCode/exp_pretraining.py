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
from dataset import Cells
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
    
    opt = parser.parse_args()

    model = CountingNetwork()
    
    norm_label = True
    
    data_train = Cells(opt.data, mode='train', norm_label=norm_label)
    data_test = Cells(opt.data, mode='validation', norm_label=norm_label)
    data_train_loader = DataLoader(data_train, batch_size=opt.batch, num_workers=opt.workers, shuffle=True)
    data_test_loader= DataLoader(data_test, batch_size=opt.batch, num_workers=opt.workers)
    
    print("TRAIN STARTED")
    criterion = nn.MSELoss()
    accuracy = AbsDic()    
    optimizer = Adam(model.parameters(), opt.lr)

    e_done = 0
    best = None

    train(model, criterion, accuracy, optimizer, data_train_loader, data_test_loader, opt.expname, epochs=opt.epochs,  logdir=opt.logs, e_done=e_done, best=best)
    
if __name__ == '__main__':
    main()
