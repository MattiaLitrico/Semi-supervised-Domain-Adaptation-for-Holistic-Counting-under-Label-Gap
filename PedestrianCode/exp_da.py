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
from dataset import Pedestrian
from network import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='Data dir')  
    parser.add_argument('--target', type=str, help='Data dir')  
    parser.add_argument('--logs', type=str, default='logs', help='Logs dir')
    parser.add_argument('--weights_fe_s', type=str, default='', help='Checkpoints dir') 
    parser.add_argument('--weights_cp', type=str, default='', help='Checkpoints dir')     
    parser.add_argument('--batch', type=int, default=32, help='Batch size')  
    parser.add_argument('--epochs', type=int, default=50, help='Epochs') 
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--workers', type=int, default=2, help='Workers')
    parser.add_argument('--expname', type=str, default='exp', help='Experiment name')
    
    opt = parser.parse_args()

    fe_s = FeatureExtractor()
    fe_t = FeatureExtractor()
    
    discriminator = Discriminator()
    counting_layers = CountingLayers(output_size=1)
    
    source_train = Pedestrian("Pedestrian/"+opt.source, mode='train')
    source_train_loader = DataLoader(source_train, batch_size=opt.batch, num_workers=opt.workers, shuffle=True)

    target_train = Pedestrian("Pedestrian/"+opt.target, mode='train', norm_label=True)
    target_test = Pedestrian("Pedestrian/"+opt.target, mode='validation', norm_label=True)
    target_train_loader = DataLoader(target_train, batch_size=opt.batch, num_workers=opt.workers, shuffle=True)
    target_test_loader= DataLoader(target_test, batch_size=opt.batch, num_workers=opt.workers)
    
    data = {'train_source': source_train_loader, 'train_target': target_train_loader,
            'test_target': target_test_loader}

    print("TRAIN STARTED") 
    
    c_loss = nn.BCEWithLogitsLoss()

    optimizer_gen_t = SGD(fe_t.parameters(), opt.lr, weight_decay=0.1)  
    optimizer_dis = SGD(discriminator.parameters(), opt.lr, weight_decay=0.1)

    e_done = 0
    best = None

    checkpoint_fe_s = load_checkpoint(opt.weights_fe_s)
    checkpoint_cp = load_checkpoint(opt.weights_cp)
    
    fe_t = load_weights(fe_t, checkpoint_fe_s)
    fe_s = load_weights(fe_s, checkpoint_fe_s)
    counting_layers = load_weights(counting_layers, checkpoint_cp)

    train_da(fe_s, fe_t, discriminator, counting_layers, c_loss, optimizer_gen_t, optimizer_dis, data, opt.expname, epochs=opt.epochs,  logdir=opt.logs, e_done=e_done,best=best)

if __name__ == '__main__':
    main()
