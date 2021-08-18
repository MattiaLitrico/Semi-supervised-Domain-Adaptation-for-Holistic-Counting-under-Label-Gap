from abc import ABCMeta,abstractmethod
import numpy as np
import pickle
import os
import dataset
import h5py

class Loader (object):
    __metaclass__ = ABCMeta

    def __init__(this,autocontrast=True,normalise=True,rescale=None,*args,**kwargs):
        this.autocontrast = autocontrast
        this.normalise = normalise
        this.rescale = rescale

    @abstractmethod
    def get_data(this,set,split=None):
        pass

    @abstractmethod
    def number_splits(this):
        pass


class CVPPP17Loader(Loader):

    def __init__(this,training_path,testing_path,a1=True,a2=True,a3=True,a4=True,a5=False,*args,**kwargs):
        super(CVPPP17Loader, this).__init__(*args, **kwargs)
        this.data_to_load = {"a1":a1,"a2":a2,"a3":a3,"a4":a4,"a5":a5}

        this.training_path = training_path
        this.testing_path = testing_path



        this.splits =  [[[1,2,3], 4,4],
                       [[2,3,4], 1,1],
                       [[3,4,1], 2,2],
                       [[4,1,2], 3,3]]


    def number_splits(this):
        return len(this.splits)

    def get_data(this,set,split=0):
        if (set=="training"):
            x_train_all, x_test_all, y_train_all, y_test_all, x_train_set, x_test_set = dataset.get_data(this.splits[split],this.training_path)
            
            filter_training = np.zeros((x_train_set.shape[0],))
            filter_testing  = np.zeros((x_test_set.shape[0],))

            for t in this.data_to_load:
                if (this.data_to_load[t]==True):
                    T = t.upper()
                    filter_training = np.logical_or(filter_training,x_train_set==T)
                    filter_testing = np.logical_or(filter_testing, x_test_set == T)

            x_train = x_train_all[filter_training,...]
            y_train = y_train_all[filter_training]
            x_test = x_test_all[filter_testing,...]
            y_test = y_test_all[filter_testing]
            
        elif (set=="testing"):
            x_train, y_train = (None, None)

            x_test_all,y_test_all,test_sets = dataset.get_data_testing(this.testing_path)

            filter = np.zeros((test_sets.shape[0],))

            for t in this.data_to_load:
                if (this.data_to_load[t] == True):
                    T = t.upper()
                    filter = np.logical_or(filter, test_sets == T)

            x_test = x_test_all[filter, ...]
            y_test = y_test_all[filter]

        else:
            raise ValueError("Set {} is not valid".format(set))

        return x_train,y_train,x_test,y_test

class MultiModalDataLoader(Loader):

    def __init__(this,path,data_file="mm_data.h5",split_file="splits.4.h5",rgb=True,ir=False,fmp=False,depth=False,*args,**kwargs):
        super(MultiModalDataLoader, this).__init__(*args, **kwargs)

        this.rgb = rgb
        this.ir=ir
        this.fmp=fmp
        this.depth=depth

        this.path = path
        this.data_file=data_file
        this.split_file= split_file

        with open(os.path.join(this.path,this.split_file),'rb') as h:
            this.splits = pickle.load(h, encoding='latin1')['splits']

    def number_splits(this):
        return len(this.splits)

    def get_data(this,set,split=0):

        with open(os.path.join(this.path,this.data_file),'rb') as h:
            data = pickle.load(h, encoding='latin1')

        xs = []

        if (this.rgb):
            rgb = data['rgb'].transpose((0,2,3,1))
            xs.append(rgb)

        if (this.ir):
            ir = data['ir'].transpose((0,2,3,1))

            xs.append(ir)

        if (this.fmp):
            fmp = data['fmp'].transpose((0,2,3,1))

            xs.append(fmp)

        if (this.depth):
            depth = data['depth'].transpose((0,2,3,1))

            xs.append(depth)

        y = data['count']

        idx_train = this.splits[split][2]
        idx_val = this.splits[split][1]
        idx_test = this.splits[split][0]

        if (set=="training"):
            x_train = [x[idx_train] for x in xs]
            x_val = [x[idx_val] for x in xs]

            y_train = y[idx_train]
            y_val = y[idx_val]

            return x_train,y_train,x_val,y_val


        elif (set=="testing"):
            x_test = [x[idx_test] for x in xs]
            y_test = y[idx_test]


            return None, None, x_test, y_test
        else:
            raise ValueError("Set {} is not valid".format(set))


class KomatsunaLoader(Loader):

    def __init__(this, filename, *args, **kwargs):
        super(KomatsunaLoader,this).__init__(*args,**kwargs)

        this.filename=filename

    def number_splits(this):
        return 1

    def get_data(this,set,split=0):
        data = h5py.File(this.filename, "r")

        X = data['X'][()]
        Y = data['Y'][()][0]
        ID = data['ID'][()][0]
        #TS = data['TS'].value[0]

        #print(TS)
        
        X = X.transpose((3,0,1,2))
        """
        if (this.autocontrast):
            X=dataset.autocontrast(np.asarray(X,dtype='uint8'))

        if (this.rescale != None):
            X=dataset.rescale(X,this.rescale)

        if (this.normalise):
            X=dataset.normalise(X)
        """
        idx_train = (ID == 0) + (ID == 1)
        idx_val = (ID == 4)
        idx_test = (ID == 2) + (ID == 3)

        if (set=="training"):
            return X[idx_train],Y[idx_train],X[idx_val],Y[idx_val]

        if (set=="testing"):
            return None, None, X[idx_test],Y[idx_test]

