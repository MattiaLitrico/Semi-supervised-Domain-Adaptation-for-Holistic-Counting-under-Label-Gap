from torch.utils.data.dataset import Dataset 
from PIL import Image, ImageOps
from os import path, listdir
from os.path import basename, splitext, join
import numpy as np
import random
import torch
from glob import glob
import scipy.io as sio
from PIL.ImageOps import autocontrast

class Cells(Dataset):
    def __init__(self, data_dir, mode, norm_label=False, normalisation="-1_1", train_perc=None):
        """
        data_dir: dir with images
        gt_dir: label dir
        train: mode train or test
        """
        self.data_dir = data_dir
        self.img_dim = (320,320)
        self.mode = mode
        self.normalise = self.__normalise_0_1__ if normalisation == "0_1" else self.__normalise__
        
        images_dir = data_dir+"/images"
        labels_dir = data_dir

        image_paths = listdir(images_dir)
        
        X = [join(images_dir, p) for p in image_paths]
        Y = self.__getlabels__(labels_dir)
        
        X = np.sort(np.array(X))
        Y = np.array(Y)
        
        np.random.seed(1234)
        idx = np.random.permutation(len(X))

        self.X = X[idx]
        self.Y = Y[idx]
        if norm_label:
            self.Y = self.__normalise_0_1__(self.Y)
        
        val_perc = 20
        test_perc = 100-val_perc-55
        
        test_len = len(self.X)*test_perc//100                   
        val_len = len(self.X)*val_perc//100
        
        if mode == 'train':
            self.X = self.X[test_len+val_len:]
            self.Y = self.Y[test_len+val_len:]
        elif mode == 'validation':
            self.X = self.X[test_len:test_len+val_len]
            self.Y = self.Y[test_len:test_len+val_len]
        else:
            self.X = self.X[:test_len]
            self.Y = self.Y[:test_len]
        
        if train_perc != None and mode == 'train':
            self.X = self.X[:train_perc]
            self.Y = self.Y[:train_perc]
        
    def __getlabels__(self, data_dir):
        file = join(data_dir, "count.mat")
        
        labels = sio.loadmat(file)["Y"]
        labels = [np.int32(l) for l in labels]

        return labels

    def __getitem__(self, index):
        """
        Return requested image
        """
        img = self.X[index]
        y = self.Y[index]
        
        x = Image.open(img)
        #Resizee 320,320
        x = x.resize(self.img_dim)
        #Autocontrast
        x = autocontrast(x)
        if self.mode == 'train':  
            #Data augmentation: flip, rotation
            x = self.__augmentation__(x)
        #Normalisation [-1,1] or [0,1]
        x = self.normalise(x)
        #To torch tensor
        x = torch.Tensor(np.transpose(x, (2, 0, 1)))
        y = torch.Tensor(y)

        return [x, y]
        
    def __len__(self):
        return len(self.X)

    def __normalise__(self, x):
        #Normalization between [-1, 1]
        x = np.array(x)

        m = x.min()
        M = x.max()

        h = (M-m)/2

        return ((x-m) / h)-1
    
    def __normalise_0_1__(self, x):
        #Normalization between [0, 1]
        x = np.array(x)

        m = x.min()
        M = x.max()

        h = (M-m)

        return ((x-m) / h)

    def __augmentation__(self, x, flip=True, rotation=range(0,360,90)):
        #Flipping and rotation
        rand = random.randint(0,10)
        if rand < 3:
            #Flipping
            if flip:
                x = x.transpose(Image.FLIP_LEFT_RIGHT)
                x = x.transpose(Image.FLIP_TOP_BOTTOM)
        elif rand > 6:
            #Rotation
            r = random.randint(0,3)
            angle = rotation[r]
            x = x.rotate(angle)

        return x