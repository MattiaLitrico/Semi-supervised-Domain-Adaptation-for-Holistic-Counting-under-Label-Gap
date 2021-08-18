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
    parser.add_argument('--weights_fe', type=str, default='', help='Checkpoints dir') 
    parser.add_argument('--weights_cp', type=str, default='', help='Checkpoints dir')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')  
    parser.add_argument('--workers', type=int, default=2, help='Workers')
    parser.add_argument('--output_size', type=int, default=1, help='Output size of the counting network')
    parser.add_argument('--train_perc', type=int, default=55, help='Percentage of training split')
    
    opt = parser.parse_args()

    model = CountingNetwork(output_size=opt.output_size)
    
    print("TEST STARTED")

    data_test = Cells(opt.data, mode='test', norm_label=False, train_perc=opt.train_perc)
    data_test_loader= DataLoader(data_test, batch_size=opt.batch, num_workers=opt.workers)

    checkpoint_fe = load_checkpoint(opt.weights_fe)    
    checkpoint_cp = load_checkpoint(opt.weights_cp) 

    load_weights(model.backbone, checkpoint_fe)
    load_weights(model.counting_part, checkpoint_cp)

    model.eval()
    mapper.eval()
    predictions, labels = test(model, data_test_loader)

    metrics = compute_metrics(labels, predictions, True)
    print_metrics(metrics)
    
if __name__ == '__main__':
    main()
