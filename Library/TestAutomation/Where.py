import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from Library.General import DataThings as DT
from Library.Network import KerasNetwork as KN


class Where(KN.KerasNetwork):
    """ regression neural network """

    def __init__(self, test):
        """ """

        # load network
        self.test = test
        KN.KerasNetwork.__init__(self, '{} where'.format(test.name),
                                 test.where_network_path) 
        self.labels = ['x1', 'y1', 'x2', 'y2']
        self.load_network()


    ### NETWORK ###

    def create_network(self, n_hidden=32, n_layers=4):
        """ create ANN reward nertwork with keras """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.test.mid_size))
        # hidden layers
        for _ in range(n_layers - 1):
            network.add(Dense(n_hidden, activation='relu'))
        # last layer
        network.add(Dense(len(self.labels)))
        network.compile(loss='mean_squared_error', optimizer='adam')
        self.network = network


    ### TRAIN - ONLINE ###  

    def train_network(self, data, actions, w, h, ep=100, cutoff=1e-8):
        """ action = (button, x, y, time)  """
        # format input data
        print('\nTraining where')
        labels = np.reshape([(a[1] / w, a[2] / h, a[3] / w, a[4] / h)
                             for a in actions], (-1, len(self.labels)))
        print(' - data shape: {}'.format(data.shape))
        print(' - labels shape: {}'.format(labels.shape))
        print(labels)
        # loop until under cutoff
        count, cost = 0, 100
        while cost > cutoff and count < 50:
            self.network.fit(data, labels, batch_size=len(labels), epochs=ep, verbose=0)
            cost = self.network.evaluate(data, labels, batch_size=len(labels), verbose=0)
            print(' - cost {}: {}'.format(count, cost))
            count += 1


