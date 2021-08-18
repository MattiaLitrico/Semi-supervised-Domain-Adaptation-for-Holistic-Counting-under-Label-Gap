import numpy as np
import os
import re
import csv
import glob
import os.path as path
import pandas as pd
from PIL import Image,ImageOps
from scipy import linalg

def get_filename(p):
    _, t = path.split(p)
    return t


def get_last_dir(p):
    h, _ = path.split(p)
    _, t = path.split(h)
    return t[:2]

def get_data(split_load, basepath='CVPPP2017_LCC_training/TrainingSplits/'):

    imgname_train_A1 = np.array([glob.glob(path.join(basepath, 'A1' + str(h) + '/*rgb.png')) for h in split_load[0]])
    imgname_train_A2 = np.array([glob.glob(path.join(basepath, 'A2' + str(h) + '/*rgb.png')) for h in split_load[0]])
    imgname_train_A3 = np.array([glob.glob(path.join(basepath, 'A3' + str(h) + '/*rgb.png')) for h in split_load[0]])
    imgname_train_A4 = np.array([glob.glob(path.join(basepath, 'A4' + str(h) + '/*rgb.png')) for h in split_load[0]])
    
    imgname_test_A1 = np.array([glob.glob(path.join(basepath, 'A1' + str(split_load[2])) + '/*rgb.png')])
    imgname_test_A2 = np.array([glob.glob(path.join(basepath, 'A2' + str(split_load[2])) + '/*rgb.png')])
    imgname_test_A3 = np.array([glob.glob(path.join(basepath, 'A3' + str(split_load[2])) + '/*rgb.png')])
    imgname_test_A4 = np.array([glob.glob(path.join(basepath, 'A4' + str(split_load[2])) + '/*rgb.png')])

    filelist_train_A1 = list(np.sort(imgname_train_A1.flat))
    filelist_train_A2 = list(np.sort(imgname_train_A2.flat))
    filelist_train_A3 = list(np.sort(imgname_train_A3.flat))
    filelist_train_A4 = list(np.sort(imgname_train_A4.flat))

    filelist_train_A1_img = np.array([np.array(get_filename(filelist_train_A1[h])) for h in range(len(filelist_train_A1))])
    filelist_train_A2_img = np.array([np.array(get_filename(filelist_train_A2[h])) for h in range(len(filelist_train_A2))])
    filelist_train_A3_img = np.array([np.array(get_filename(filelist_train_A3[h])) for h in range(len(filelist_train_A3))])
    filelist_train_A4_img = np.array([np.array(get_filename(filelist_train_A4[h])) for h in range(len(filelist_train_A4))])

    filelist_train_A1_set = np.array([np.array(get_last_dir(filelist_train_A1[h])) for h in range(len(filelist_train_A1))])
    filelist_train_A2_set = np.array([np.array(get_last_dir(filelist_train_A2[h])) for h in range(len(filelist_train_A2))])
    filelist_train_A3_set = np.array([np.array(get_last_dir(filelist_train_A3[h])) for h in range(len(filelist_train_A3))])
    filelist_train_A4_set = np.array([np.array(get_last_dir(filelist_train_A4[h])) for h in range(len(filelist_train_A4))])
   
    filelist_test_A1 = list(np.sort(imgname_test_A1.flat))
    filelist_test_A2 = list(np.sort(imgname_test_A2.flat))
    filelist_test_A3 = list(np.sort(imgname_test_A3.flat))
    filelist_test_A4 = list(np.sort(imgname_test_A4.flat))
    filelist_test_A1_img = np.array([np.array(get_filename(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_img = np.array([np.array(get_filename(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_img = np.array([np.array(get_filename(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_img = np.array([np.array(get_filename(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])
    filelist_test_A1_set = np.array([np.array(get_last_dir(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_set = np.array([np.array(get_last_dir(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_set = np.array([np.array(get_last_dir(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_set = np.array([np.array(get_last_dir(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])

    # Read image names into np array train
    x_train_A1 = np.array(filelist_train_A1)
    x_train_A2 = np.array(filelist_train_A2)
    x_train_A3 = np.array(filelist_train_A3)
    x_train_A4 = np.array(filelist_train_A4)
    
    # Read image names into np array test
    x_test_A1 = np.array(filelist_test_A1)
    x_test_A2 = np.array(filelist_test_A2)
    x_test_A3 = np.array(filelist_test_A3)
    x_test_A4 = np.array(filelist_test_A4)

    x_train_all = np.concatenate((x_train_A1, x_train_A2, x_train_A3, x_train_A4), axis=0)
    x_test_all = np.concatenate((x_test_A1, x_test_A2, x_test_A3, x_test_A4), axis=0)

    x_train_set = np.concatenate(
        (filelist_train_A1_set, filelist_train_A2_set, filelist_train_A3_set, filelist_train_A4_set), axis=0)
    x_test_set = np.concatenate((filelist_test_A1_set, filelist_test_A2_set, filelist_test_A3_set, filelist_test_A4_set),
                                axis=0)

    ###############################
    # Getting targets (y data)    #
    ###############################
    counts_A1 = np.array([glob.glob(path.join(basepath, 'A1.xlsx'))])
    counts_A2 = np.array([glob.glob(path.join(basepath, 'A2.xlsx'))])
    counts_A3 = np.array([glob.glob(path.join(basepath, 'A3.xlsx'))])
    counts_A4 = np.array([glob.glob(path.join(basepath, 'A4.xlsx'))])

    counts_train_flat_A1 = list(counts_A1.flat)
    train_labels_A1 = pd.DataFrame()
    y_train_A1_list = []
    y_val_A1_list = []
    y_test_A1_list = []
    for f in counts_train_flat_A1:
        frame = pd.read_excel(f, header=None)
        train_labels_A1 = train_labels_A1.append(frame, ignore_index=False)
    all_labels_A1 = np.array(train_labels_A1)

    for j in filelist_train_A1_img:
        arr_idx = np.where(all_labels_A1 == j)
        y_train_A1_list.append(all_labels_A1[arr_idx[0], :])
    y_train_A1_labels = np.concatenate(y_train_A1_list, axis=0)

    for j in filelist_test_A1_img:
        arr_idx = np.where(all_labels_A1 == j)
        y_test_A1_list.append(all_labels_A1[arr_idx[0], :])
    y_test_A1_labels = np.concatenate(y_test_A1_list, axis=0)

    counts_train_flat_A2 = list(counts_A2.flat)
    train_labels_A2 = pd.DataFrame()
    y_train_A2_list = []
    y_val_A2_list = []
    y_test_A2_list = []
    for f in counts_train_flat_A2:
        frame = pd.read_excel(f, header=None)
        train_labels_A2 = train_labels_A2.append(frame, ignore_index=False)
    all_labels_A2 = np.array(train_labels_A2)

    for j in filelist_train_A2_img:
        arr_idx = np.where(all_labels_A2 == j)
        y_train_A2_list.append(all_labels_A2[arr_idx[0], :])
    y_train_A2_labels = np.concatenate(y_train_A2_list, axis=0)

    for j in filelist_test_A2_img:
        arr_idx = np.where(all_labels_A2 == j)
        y_test_A2_list.append(all_labels_A2[arr_idx[0], :])
    y_test_A2_labels = np.concatenate(y_test_A2_list, axis=0)
    
    counts_train_flat_A3 = list(counts_A3.flat)
    train_labels_A3 = pd.DataFrame()
    y_train_A3_list = []
    y_val_A3_list = []
    y_test_A3_list = []
    for f in counts_train_flat_A3:
        frame = pd.read_excel(f, header=None)
        train_labels_A3 = train_labels_A3.append(frame, ignore_index=False)
    all_labels_A3 = np.array(train_labels_A3)

    for j in filelist_train_A3_img:
        arr_idx = np.where(all_labels_A3 == j)
        y_train_A3_list.append(all_labels_A3[arr_idx[0], :])
    y_train_A3_labels = np.concatenate(y_train_A3_list, axis=0)

    for j in filelist_test_A3_img:
        arr_idx = np.where(all_labels_A3 == j)
        y_test_A3_list.append(all_labels_A3[arr_idx[0], :])
    y_test_A3_labels = np.concatenate(y_test_A3_list, axis=0)
    
    counts_train_flat_A4 = list(counts_A4.flat)
    train_labels_A4 = pd.DataFrame()
    y_train_A4_list = []
    y_val_A4_list = []
    y_test_A4_list = []
    for f in counts_train_flat_A4:
        frame = pd.read_excel(f, header=None)
        train_labels_A4 = train_labels_A4.append(frame, ignore_index=False)
    all_labels_A4 = np.array(train_labels_A4)

    for j in filelist_train_A4_img:
        arr_idx = np.where(all_labels_A4 == j)
        y_train_A4_list.append(all_labels_A4[arr_idx[0], :])
    y_train_A4_labels = np.concatenate(y_train_A4_list, axis=0)

    for j in filelist_test_A4_img:
        arr_idx = np.where(all_labels_A4 == j)
        y_test_A4_list.append(all_labels_A4[arr_idx[0], :])
    y_test_A4_labels = np.concatenate(y_test_A4_list, axis=0)

    y_train_all_labels = np.concatenate((y_train_A1_labels, y_train_A2_labels, y_train_A3_labels, y_train_A4_labels),
                                        axis=0)
    y_test_all_labels = np.concatenate((y_test_A1_labels, y_test_A2_labels, y_test_A3_labels, y_test_A4_labels), axis=0)

    y_train_all = y_train_all_labels[:, 1]
    y_test_all = y_test_all_labels[:, 1]


    return x_train_all, x_test_all, y_train_all, y_test_all, x_train_set, x_test_set

def get_data_testing(basepath='CVPPP2017_testing/testing'):
    ###############################
    # Getting images (x data)	  #
    ###############################

    imgname_test_A1 = np.array([glob.glob(path.join(basepath,'A1','*rgb.png'))])
    imgname_test_A2 = np.array([glob.glob(path.join(basepath,'A2','*rgb.png'))])
    imgname_test_A3 = np.array([glob.glob(path.join(basepath,'A3','*rgb.png'))])
    imgname_test_A4 = np.array([glob.glob(path.join(basepath,'A4','*rgb.png'))])
    imgname_test_A5 = np.array([glob.glob(path.join(basepath,'A5','*rgb.png'))])

    filelist_test_A1 = list(np.sort(imgname_test_A1.flat))
    filelist_test_A2 = list(np.sort(imgname_test_A2.flat))
    filelist_test_A3 = list(np.sort(imgname_test_A3.flat))
    filelist_test_A4 = list(np.sort(imgname_test_A4.flat))
    filelist_test_A5 = list(np.sort(imgname_test_A5.flat))

    filelist_test_A1_img = np.array([np.array(get_filename(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_img = np.array([np.array(get_filename(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_img = np.array([np.array(get_filename(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_img = np.array([np.array(get_filename(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])
    filelist_test_A5_img = np.array([np.array(get_filename(filelist_test_A5[h])) for h in range(len(filelist_test_A5))])
    filelist_test_A1_set = np.array([np.array(get_last_dir(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_set = np.array([np.array(get_last_dir(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_set = np.array([np.array(get_last_dir(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_set = np.array([np.array(get_last_dir(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])
    filelist_test_A5_set = np.array([np.array(get_last_dir(filelist_test_A5[h])) for h in range(len(filelist_test_A5))])

    x_test_img = np.concatenate((filelist_test_A1_img, filelist_test_A2_img, filelist_test_A3_img, filelist_test_A4_img,filelist_test_A5_img), axis=0)
    
    x_test_A1 = np.array(filelist_test_A1)
    x_test_A2 = np.array(filelist_test_A2)
    x_test_A3 = np.array(filelist_test_A3)
    x_test_A4 = np.array(filelist_test_A4)
    x_test_A5 = np.array(filelist_test_A5)
    
    ###############################
    # Getting targets (y data)	  #
    ###############################
    counts_A1 = np.array([glob.glob(path.join(basepath, 'A1/A1.xlsx'))])
    counts_A2 = np.array([glob.glob(path.join(basepath, 'A2/A2.xlsx'))])
    counts_A3 = np.array([glob.glob(path.join(basepath, 'A3/A3.xlsx'))])
    counts_A4 = np.array([glob.glob(path.join(basepath, 'A4/A4.xlsx'))])
    counts_A5 = np.array([glob.glob(path.join(basepath, 'A5/A5.xlsx'))])

    counts_train_flat_A1 = list(counts_A1.flat)
    train_labels_A1 = pd.DataFrame()
    y_test_A1_list = []
    for f in counts_train_flat_A1:
        frame = pd.read_excel(f, header=None)
        train_labels_A1 = train_labels_A1.append(frame, ignore_index=False)
    all_labels_A1 = np.array(train_labels_A1)

    for j in filelist_test_A1_img:
        arr_idx = np.where(all_labels_A1 == j)
        y_test_A1_list.append(all_labels_A1[arr_idx[0], :])
    y_test_A1_labels = np.concatenate(y_test_A1_list, axis=0)

    counts_train_flat_A2 = list(counts_A2.flat)
    train_labels_A2 = pd.DataFrame()
    y_test_A2_list = []
    for f in counts_train_flat_A2:
        frame = pd.read_excel(f, header=None)
        train_labels_A2 = train_labels_A2.append(frame, ignore_index=False)
    all_labels_A2 = np.array(train_labels_A2)


    for j in filelist_test_A2_img:
        arr_idx = np.where(all_labels_A2 == j)
        y_test_A2_list.append(all_labels_A2[arr_idx[0], :])
    y_test_A2_labels = np.concatenate(y_test_A2_list, axis=0)

    counts_train_flat_A3 = list(counts_A3.flat)
    train_labels_A3 = pd.DataFrame()
    y_test_A3_list = []
    for f in counts_train_flat_A3:
        frame = pd.read_excel(f, header=None)
        train_labels_A3 = train_labels_A3.append(frame, ignore_index=False)
    all_labels_A3 = np.array(train_labels_A3)


    for j in filelist_test_A3_img:
        arr_idx = np.where(all_labels_A3 == j)
        y_test_A3_list.append(all_labels_A3[arr_idx[0], :])
    y_test_A3_labels = np.concatenate(y_test_A3_list, axis=0)

    counts_train_flat_A4 = list(counts_A4.flat)
    train_labels_A4 = pd.DataFrame()
    y_test_A4_list = []
    for f in counts_train_flat_A4:
        frame = pd.read_excel(f, header=None)
        train_labels_A4 = train_labels_A4.append(frame, ignore_index=False)
    all_labels_A4 = np.array(train_labels_A4)

    for j in filelist_test_A4_img:
        arr_idx = np.where(all_labels_A4 == j)
        y_test_A4_list.append(all_labels_A4[arr_idx[0], :])
    y_test_A4_labels = np.concatenate(y_test_A4_list, axis=0)


    #################
    counts_train_flat_A5 = list(counts_A5.flat)
    train_labels_A5 = pd.DataFrame()
    y_test_A5_list = []
    for f in counts_train_flat_A5:
        frame = pd.read_excel(f, header=None)
        train_labels_A5 = train_labels_A5.append(frame, ignore_index=False)
    all_labels_A5 = np.array(train_labels_A5)

    for j in filelist_test_A5_img:
        arr_idx = np.where(all_labels_A5 == j)
        y_test_A5_list.append(all_labels_A5[arr_idx[0], :])
    y_test_A5_labels = np.concatenate(y_test_A5_list, axis=0)
    #################

    x_test_all = np.concatenate((x_test_A1, x_test_A2, x_test_A3, x_test_A4, x_test_A5), axis=0)
    y_test_all = np.concatenate((y_test_A1_labels, y_test_A2_labels, y_test_A3_labels, y_test_A4_labels,y_test_A5_labels),axis=0)
    test_sets  = np.concatenate((filelist_test_A1_set,filelist_test_A2_set,filelist_test_A3_set,filelist_test_A4_set,filelist_test_A5_set))

    y_test_all = y_test_all[:, 1]

    return x_test_all,y_test_all,test_sets








