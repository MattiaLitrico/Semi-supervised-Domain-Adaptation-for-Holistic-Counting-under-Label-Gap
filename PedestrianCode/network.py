from torch.utils.data import DataLoader
from models import *
from torchvision import transforms
from torch.optim import SGD, Adam
import os
from os.path import join, basename, splitext
from metrics import *
import torch
import numpy as np
from PIL import Image
from torch import nn
import argparse
import time
import copy

class AverageValueMeter():
    def __init__(self, best=None, mode='best'):
        self.reset()
        self.best_value = best
        self.mode = mode
    def reset(self):
        self.sum = 0
        self.num = 0
    def add(self, value, num, mode):
        self.sum += value*num
        self.num += num

    def is_best(self, value):
        if self.best_value == None:
            self.best_value = value
            return True
        if self.mode == 'best' and value < self.best_value:
            self.best_value = value
            return True
        if self.mode == 'worst' and value > self.best_value:
            self.best_value = value
            return True
        
        return False
    
    def value(self):
        try:
            return self.sum/self.num
        except:
            return None

def train_da(generator_s, generator_t, discriminator, counting_layers, c_loss, kld, optimizer_gen, optimizer_dis, optimizer_kld, data, exp_name='experiment', epochs=50, logdir='logs', e_done=0, best=None):
    
    mmd = MMD_loss()
    sigmoid = nn.Sigmoid()
    
    #Try to create a dir where we save weights
    try:
        os.makedirs(join(logdir, exp_name))
    except:
        pass
    
    generator_iterations = 1
    models = [generator_s, generator_t, discriminator, counting_layers]

    acc_meter = AverageValueMeter()
    loss_meter = AverageValueMeter()
    loss_meter_kld = AverageValueMeter()
    loss_meter_gen = AverageValueMeter()
    loss_meter_real = AverageValueMeter()
    loss_meter_fake = AverageValueMeter()
    loss_meter_count = AverageValueMeter()
    
    absdic = AbsDic()
    reg = VarianceRegularizer(lambdaint=0.01)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    global_step = 0
    not_improving = 0

    for model in models:
        model.to(device)
    
    generator_s.eval()
    generator_t.train()
    discriminator.train()
    counting_layers.eval()
    
    model = CountingNetwork(output_size=1)
    model.counting_part = counting_layers
    for e in range(e_done, epochs):
        
        list_y_real = list()
        list_y_fake = list()
        list_mmd = list()
        
        te = time.time()
        mode = 'train'
        
        loss_meter_kld.reset()
        loss_meter_gen.reset()
        loss_meter_real.reset()
        loss_meter_fake.reset()
        loss_meter_count.reset()
        acc_meter.reset()
        loss_meter.reset()

        set_requires_grad(generator_s, requires_grad=False)
        set_requires_grad(counting_layers, requires_grad=False)

        generator_s.apply(set_bn_eval)
        counting_layers.apply(set_bn_eval)

        for i, (batch_s, batch_t) in enumerate(zip(data['train'+'_source'],data[mode+'_target'])):
            t1 = time.time()

            x_s = batch_s[0].to(device) 
            x_t = batch_t[0].to(device)
            y_t = batch_t[1].to(device)

            n = x_s.shape[0] #num of elements in batch

            y_fake = ((torch.rand(n, 1) - 0.0)*0.3 + 0.0).to(device)
            y_real = ((torch.rand(n, 1) - 0.0)*0.5 + 0.7).to(device)
            #Train discriminator
            set_requires_grad(generator_t, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)

            output_gen_s = generator_s(x_s)
            output_gen_t = generator_t(x_t)
            output_dis_s = discriminator(output_gen_s)
            output_dis_t = discriminator(output_gen_t)

            l_real = c_loss(output_dis_s, y_real)
            l_fake = c_loss(output_dis_t, y_fake)
            l_dis = (l_fake + l_real)
                        
            l_dis.backward()
            optimizer_dis.step()
            optimizer_dis.zero_grad()
            
            #To analyse
            list_y_real.append(np.abs(sigmoid(output_dis_s).detach().cpu().numpy()-0.5).mean())
            list_y_fake.append(np.abs(sigmoid(output_dis_t).detach().cpu().numpy()-0.5).mean())
            list_mmd.append(mmd(output_gen_s,output_gen_t).detach().cpu().numpy())

            #Train generator
            set_requires_grad(generator_t, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)        
                        
            for it in range(generator_iterations):
                output_gen_t = generator_t(x_t)
                output_dis_t = discriminator(output_gen_t)
                output_counting_t = counting_layers(output_gen_t)
                
                regularizer = reg(output_counting_t)
                l_gen = c_loss(output_dis_t, y_real) + regularizer
                
                l_gen.backward(retain_graph=False)
                optimizer_gen.step()
                optimizer_gen.zero_grad()
                         
                absdic_t = absdic(y_t-output_counting_t)[0]

            
            loss_meter_kld.add(regularizer.item(), n, mode)
            loss_meter_gen.add(l_gen.item(), n, mode)
            loss_meter_real.add(l_real.item(), n, mode)
            loss_meter_fake.add(l_fake.item(), n, mode)
            loss_meter_count.add(absdic_t, 1, mode)
            
            print("{}: Epochs {}/{}: [{}/{}] ->->-> Elapsed Time {:.2f}/{:.2f} secs --- Loss_Real {:.4f}, Loss_Fake {:.4f}, Loss_gen {:.4f}, Loss_KLD {:.4f}, |DIC| {:.4f} \n".format(mode, e, epochs, i, len(data[mode+'_source']), (time.time() - te), (time.time() - t1)*len(data[mode+'_source']), loss_meter_real.value(), loss_meter_fake.value(), loss_meter_gen.value(), loss_meter_kld.value(), loss_meter_count.value()))
            
                
        # Test phase
        generator_t = set_requires_grad(generator_t, requires_grad=False)
        model.backbone = copy.deepcopy(generator_t)
        model.backbone = set_requires_grad(model.backbone, requires_grad=False)
        
        model.apply(set_bn_eval)
        
        model.eval()

        predictions, labels = test(model, data['test_target']) 
        metrics = compute_metrics(labels, predictions, round=False)
        print_metrics(metrics)
        mse = metrics['mse']
        abs_dic = metrics['abs_dic']
                            
        acc_meter.add(abs_dic[0], 32, 'test')
        
        list_y_real = np.array(list_y_real).mean()
        list_y_fake = np.array(list_y_fake).mean()
        list_mmd = np.array(list_mmd).mean()
        
        #Save weights of generator
        save_weights(generator_t, e, optimizer_gen, acc_meter.best_value, join(logdir,exp_name) + '/%s.tar'%(exp_name+'_gen'))
        
        if acc_meter.is_best(np.array([list_y_real, list_y_fake]).mean()) and loss_meter.is_best(list_mmd):
            save_weights(generator_t, e, optimizer_gen, acc_meter.best_value, join(logdir,exp_name) + '/%s_best.tar'%(exp_name+'_gen'))
            save_weights(discriminator, e, optimizer_dis, acc_meter.best_value, join(logdir,exp_name) + '/%s_best.tar'%(exp_name+'_dis'))
            not_improving = 0
            
            print("THIS IS THE BEST \n")
        else:
            not_improving += 1
        
        #Save weights of discriminator
        save_weights(discriminator, e, optimizer_dis, acc_meter.best_value, join(logdir,exp_name) + '/%s.tar'%(exp_name+'_dis'))
        
        print("LOSS_DIS: Mean {}, std {}\n".format(np.array([list_y_real,list_y_fake]).mean(),0))
        print("LOSS_MMD: Mean {}, std {}\n".format(np.mean(list_mmd),np.std(list_mmd)))
        
        if not_improving > 30:
            return
    return model

def train(model, criterion, accuracy, optimizer, train_loader, test_loader, exp_name='experiment', epochs=50, logdir='logs', e_done=0, best=None):
    #Try to create a dir where we save weights
    try:
        os.makedirs(join(logdir, exp_name))
    except:
        pass

    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter(best)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    global_step = 0

    model.to(device)

    loader = {
        'train' : train_loader,
        'test' : test_loader,
    }

    for e in range(e_done, epochs):
        te = time.time()
        
        for mode in ['train', 'test']:
            loss_meter.reset()
            acc_meter.reset()
            
            model.train() if mode == 'train' else model.eval()
            
            with torch.set_grad_enabled(mode == 'train'):
                for i, batch in enumerate(loader[mode]):                    
                    t1 = time.time()

                    x = batch[0].to(device) 
                    y = batch[1].to(device)

                    output = model(x)
                    
                    if output.size()[1] > 1: #e.g. multi-output mode
                        output = torch.sum(output, dim=1, keepdim=True)
                    
                    n = x.shape[0] #num of elements in batch
                    global_step += n
                    
                    l = criterion(output.float(), y.float())

                    if mode == 'train':
                        l.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    #acc = accuracy(y.to('cpu')-torch.round(output.to('cpu')))
                    acc = accuracy(y.to('cpu')-output.to('cpu'))

                    loss_meter.add(l.item(), n, mode)
                    acc_meter.add(acc[0] ,n, mode)
                
                    print("{}: Epochs {}/{}: [{}/{}] ->->-> Elapsed Time {:.2f}/{:.2f} secs --- Loss {:.4f}, Acc {:.4f} \n".format(mode, e, epochs, i, len(loader[mode]), (time.time() - te), (time.time() - t1)*len(loader[mode]), loss_meter.value(), acc_meter.value()))
        
        #Save weights of feature extractor
        save_weights(model.backbone, e, optimizer, acc_meter.best_value, join(logdir,exp_name) + '/%s.tar'%(exp_name+'_fe'))
        
        if acc_meter.is_best(loss_meter.value()):
            save_weights(model.backbone, e, optimizer, acc_meter.best_value, join(logdir,exp_name) + '/%s_best.tar'%(exp_name+'_fe'))
            save_weights(model.counting_part, e, optimizer, acc_meter.best_value, join(logdir,exp_name) + '/%s_best.tar'%(exp_name+'_cp'))
        
        #Save weights of counting network
        save_weights(model.counting_part, e, optimizer, acc_meter.best_value, join(logdir,exp_name) + '/%s.tar'%(exp_name+'_cp'))
            
        
    return model


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()

def train_regressor(model, criterion, accuracy, optimizer, train_loader, test_loader, exp_name='experiment', epochs=50, logdir='logs', e_done=0, best=None):
    #Try to create a dir where we save weights
    try:
        os.makedirs(join(logdir, exp_name))
    except:
        pass

    loss_meter = AverageValueMeter(best)
    acc_meter = AverageValueMeter(best)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    global_step = 0
    not_improving = 0

    model.to(device)
    model.backbone.eval()
    
    loader = {
        'train' : train_loader,
        'test' : test_loader,
    }

    for e in range(e_done, epochs):
        te = time.time()
        
        for mode in ['train', 'test']:
            loss_meter.reset()
            acc_meter.reset()
            
            model.counting_part.train() if mode == 'train' else model.counting_part.eval()
            
            set_requires_grad(model.backbone, requires_grad=False)
            model.backbone.apply(set_bn_eval)
            for i, batch in enumerate(loader[mode]):
                t1 = time.time()

                x = batch[0].to(device) 
                y = batch[1].to(device)

                output = model(x)

                n = x.shape[0] #num of elements in batch
                global_step += n
                
                l = criterion(output.float(), y.float())

                if mode == 'train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                acc = accuracy(y.to('cpu')-torch.round(output.to('cpu')))

                loss_meter.add(l.item(), n, mode)
                acc_meter.add(acc[0] ,n, mode)

                print("{}: Epochs {}/{}: [{}/{}] ->->-> Elapsed Time {:.2f}/{:.2f} secs --- Loss {:.4f}, Acc {:.4f} \n".format(mode, e, epochs, i, len(loader[mode]), (time.time() - te), (time.time() - t1)*len(loader[mode]), loss_meter.value(), acc_meter.value()))
        
        not_improving += 1
        #Save weights of feature extractor
        save_weights(model.counting_part, e, optimizer, loss_meter.best_value, join(logdir,exp_name) + '/%s.tar'%(exp_name))
        
        if loss_meter.is_best(loss_meter.value()):
            print("SAVING BEST")
            save_weights(model.counting_part, e, optimizer, acc_meter.best_value, join(logdir,exp_name) + '/%s_best.tar'%(exp_name))
            not_improving = 0
        
        if not_improving > 60:
            return
        
    return model

def test(model, loader):
    device ="cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    
    model.eval()
    
    predictions, labels = [], []
    
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(loader):

            print("Processing batch:{}/{}".format( i, len(loader)))
            x = batch[0].to(device)
            y = batch[1].to(device)

            output = model(x)

            if output.size()[1] > 1: #e.g. multi-output mode
                output = torch.sum(output, dim=1, keepdim=True)

            preds = output.to('cpu').numpy()
            labs = y.to('cpu').numpy()

            predictions.extend(list(preds))
            labels.extend(list(labs))

    return np.array(predictions), np.array(labels)

def load_checkpoint(weights, cpu=False):
    if not cpu:
        checkpoint = torch.load(weights)
    else:
        checkpoint = torch.load(weights, map_location=torch.device('cpu'))
    
    return checkpoint

def load_weights(model, checkpoint):
    model.load_state_dict(checkpoint['weights'])

    return model

def load_epochs_done(checkpoint):
    return checkpoint['epochs'] + 1

def load_best_value(checkpoint):
    try:
        return checkpoint['best']
    except:
        return None

def load_optimizer(optimizer, checkpoint):
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return optimizer
    except:
        return optimizer

def save_weights(model, e, optimizer, best, filename):
    torch.save({
            'epochs': e,
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best': best}, filename)
    
def unnormalize_pred(predictions, min_val, max_val):
    return (predictions * (max_val-min_val)) + min_val