import os
import time
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

from Library.General import DataThings as DT
from Library.Network import Networks
from Library.Network import Autoencoder


### CALLABLE ###

new_auto = Autoencoder.create
load_auto = Networks.NetworkAuto


### TRAIN ###

def train_auto(network, subdata, h, w, n_train=100, alpha=0.0001, n_plot=20,
               pad=4, kmax_cost=10, kmax_img=10, kmax_save=0, randomize=False):
    """ iterate through data set to train autoencoder newtork """
    costs = []
    for k in range(n_train):
        #subdata = DT.subdata(DT.pad_me(subdata, pad, pad), h, w)
        network.train_network(subdata, alpha)
        costs = check_cost(network, subdata, subdata, costs, k, kmax_cost)
        check_auto(network, subdata, h, w, n_plot, randomize, k, kmax_img)
        check_save(network, k, kmax_save)


### GENERAL NETWORK ###

def check_cost(network, input_data, output_data, costs, k, k_max):
    """ run every n iterations to check network cost  """
    if k % k_max == 0:
        costs, m = get_cost_slope(network, input_data, output_data, costs)
        print('Cost {} {:.7f} {:.7f}'.format(k, costs[-1], m))
    return costs

def check_auto(network, input_data, h, w, n_plot, k, k_max):
    """ if desired, run every n iterations to plot data """
    if k_max != 0 and k % k_max == 0:
        plot_middle(network, input_data, h, w, n_plot, count=k)

def check_save(network, k, k_max):
    """ if desired, run every n iterations to save network """
    if k * k_max != 0 and k % k_max == 0:
        network.save_network(step=k)


### HELPER ###

def get_cost_slope(network, data, labels, costs):
    """ returns updates cost list and calculate slope """
    costs.append(network.get_cost(data, labels)
                 if data.shape != labels.shape else network.get_cost(data))
    if len(costs) > 1:
        costs = costs[1:] if len(costs) > 30 else costs
        avg, _ = np.polyfit(range(len(costs)), costs, 1)
        return costs, avg
    return costs, 0

def get_subdata(data, n_plot, randomize):
    """ subdata of first n data points or random data points """
    if randomize:
        idxs = random.sample(range(len(data)), n_plot)
    else:
        idxs = list(range(n_plot))
    return np.array([data[i] for i in idxs])

def plot_middle(network, input_data, h, w, n_plot, count=None, n_x=3, n_y=6):
    """ plot middle layer for autoencoder network """
    # plot inital data
    data = get_subdata(input_data, n_plot, False)
    mids = network.get_flat(data)
    outs = np.clip(network.get_outputs(data), 0, 1)
    both = [[data[i], outs[i], np.reshape(g, network.mid_size)]
            for i, g in enumerate(mids)]
    both = list(itertools.chain.from_iterable(both))
    save_path = 'plt_mid_{}'.format(count) if count or count == 0 else False
    DT.plot_data_multiple(both, save_path=save_path, n_x=n_x, n_y=n_y)
    # plot random data
    data = get_subdata(input_data, n_plot, True)
    mids = network.get_flat(data)
    outs = np.clip(network.get_outputs(data), 0, 1)
    both = [[data[i], outs[i], np.reshape(g, network.mid_size)]
            for i, g in enumerate(mids)]
    both = list(itertools.chain.from_iterable(both))
    save_path = 'plt_rdm_{}'.format(count) if count or count == 0 else False
    DT.plot_data_multiple(both, save_path=save_path, n_x=n_x, n_y=n_y)


