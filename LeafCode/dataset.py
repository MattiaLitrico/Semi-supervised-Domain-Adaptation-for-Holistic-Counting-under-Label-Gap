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
from loader import *
from leaf_dataset_utils import *

class ImageTransformation():
    def normalise(self, x):
        #Normalization between [-1, 1]
        x = np.array(x)

        m = x.min()
        M = x.max()

        h = (M-m)/2

        return ((x-m) / h)-1

    def normalise_0_1__(self, x):
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


class CVPPP2017(Dataset, ImageTransformation):
    def __init__(self, data_dir, mode, split=0, norm_label=False):
        training_path = join(data_dir,"CVPPP_2017_training_splits/TrainingSplits") if (mode=="train" or mode=="validation") else ""
        testing_path = join(data_dir,"CVPPP2017_testing/testing") if mode=="test" else ""
        set = 'training' if (mode=="train" or mode=="validation") else "testing"
        
        self.img_dim = (320,320)
        self.mode = mode

        loader = CVPPP17Loader(training_path=training_path,testing_path=testing_path, a3=False, a5=False, autocontrast=True,normalise=True)
        X_train, Y_train, X_test, Y_test = loader.get_data(set, split)

        self.X, self.Y = (X_train, Y_train) if mode=="train" else (X_test, Y_test)
        
        if norm_label:
            self.Y = self.normalise_0_1__(self.Y)

    def __getitem__(self, index):
        """
        Return requested image
        """
        img = self.X[index]
        y = self.Y[index]

        x = Image.open(img)
        #RGBA to RGB
        x = x.convert(mode="RGB")
        #Resizee 320,320
        x = x.resize(self.img_dim)
        #Autocontrast
        x = autocontrast(x)
        
        if self.mode == 'train':  
            #Data augmentation: flip, rotation
            x = self.__augmentation__(x)
        #Normalisation [-1,1]
        x = self.normalise(x)
        #To torch tensor
        x = torch.Tensor(np.transpose(x, (2, 0, 1)))
        y = torch.tensor(y).view(-1)
        
        return [x, y]
    
    def __len__(self):
        return len(self.X)

class MultiModal(Dataset, ImageTransformation):
    def __init__(self, data_dir, mode, split=0, norm_label=False, train_perc=None):
        set = 'training' if (mode=="train" or mode=="validation") else "testing"
        
        self.img_dim = (320,320)
        self.mode = mode

        loader = MultiModalDataLoader(path=data_dir)
        X_train, Y_train, X_test, Y_test = loader.get_data(set)
        self.X, self.Y = (X_train, Y_train) if mode=="train" else (X_test, Y_test)
        self.X = self.X[0] #change dimensions from (1,288,119,119,3) to (288,119,119,3)
        
        if norm_label:
            self.Y = self.normalise_0_1__(self.Y)

        if train_perc != None and mode == "train":
            np.random.seed(1234)
            idx = np.random.permutation(len(self.X))

            self.X = self.X[idx]
            self.Y = self.Y[idx]
            
            self.X = self.X[:train_perc]
            self.Y = self.Y[:train_perc]
  

    def __getitem__(self, index):
        """
        Return requested image
        """
        img = self.X[index]
        y = self.Y[index]

        img = np.asarray(img,dtype='uint8')
        x = Image.fromarray(img)        
        #RGBA to RGB
        x = x.convert(mode="RGB")
        #Resizee 320,320
        x = x.resize(self.img_dim)
        #Autocontrast
        x = autocontrast(x)
        
        if self.mode == 'train':  
            #Data augmentation: flip, rotation
            x = self.__augmentation__(x)
        #Normalisation [-1,1]
        x = self.normalise(x)
        #To torch tensor
        x = torch.Tensor(np.transpose(x, (2, 0, 1)))
        y = torch.tensor(y).view(-1)
        
        return [x, y]
    
    def __len__(self):
        return len(self.X)

class Komatsuna(Dataset, ImageTransformation):
    def __init__(self, data_dir, mode, norm_label=False, train_perc=None):
        set = 'training' if (mode=="train" or mode=="validation") else "testing"
        
        self.img_dim = (320,320)
        self.mode = mode

        loader = KomatsunaLoader(filename=data_dir+"/komatsuna.mat")
        X_train, Y_train, X_test, Y_test = loader.get_data(set)
        self.X, self.Y = (X_train, Y_train) if mode=="train" else (X_test, Y_test)
        
        if norm_label:
            self.Y = self.normalise_0_1__(self.Y)
        
        if train_perc != None and mode == "train":
            np.random.seed(1234)
            idx = np.random.permutation(len(self.X))

            self.X = self.X[idx]
            self.Y = self.Y[idx]
            
            self.X = self.X[:train_perc]
            self.Y = self.Y[:train_perc]
        print(len(self.X))
            
    def __getitem__(self, index):
        """
        Return requested image
        """
        img = self.X[index]
        y = self.Y[index]

        img = np.asarray(img,dtype='uint8')

        x = Image.fromarray(img)
        #RGBA to RGB
        x = x.convert(mode="RGB")
        #Resizee 320,320
        x = x.resize(self.img_dim)
        #Autocontrast
        x = autocontrast(x)
        
        if self.mode == 'train':  
            #Data augmentation: flip, rotation
            x = self.__augmentation__(x)
        #Normalisation [-1,1]
        x = self.normalise(x)
        #To torch tensor
        x = torch.Tensor(np.transpose(x, (2, 0, 1)))
        y = torch.tensor(y).view(-1)
        
        return [x, y]
    
    def __len__(self):
        return len(self.X)