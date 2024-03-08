import os
import time
import numpy as np

import paths
from Library.General import DataThings as DT
from Library.Network import NetworkAPI as NETS


### TRAIN ###

def train_auto_network_online():
    """ DO ME """
    done = False
    while not done:
        NETS.train_auto(auto_network, Screen.get_data_resized(w, h), h, w,
                        n_train=5, kmax_img=n_train*2, kmax_cost=2)
        time.sleep(0.5)
        done = True

def train_auto_offline():
    """ train auto network given data set """
    NETS.train_auto(auto_network, DT.load_images(paths.auto_imgs_path), h, w,
                    n_train=120, kmax_img=5, kmax_cost=2, alpha=0.0001)


### PARAMS ###

# data input size
h = 1024
w = 1024
batch = 4


### PROGRAM ###

# CREATE
if 1:
    print('Creating AUTO...')
    subname = 'test_auto'
    hidden = [16,16,16,16,16,16,16,16]
    NETS.new_auto(paths.network_path, subname, h, w, hidden, batch_size=batch)


# LOAD
if 0:
    print('Loading AUTO...')
    name = 'AUTO_testing_1024_1024_8_256'
    auto_network = DT.load_auto(paths.network_path, name)
    

# TRAIN
if 0:
    print('Training AUTO...')
    train_auto_offline()


# TEST
if 0:
    print('Testing AUTO...')
    ds = DT.load_images(paths.auto_imgs_path, n_files=10)
    NETS.plot_middle(auto_network, ds, h, w, 2, count=100, n_x=2, n_y=3)


