import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from Library.General import DataThings as DT
from Library.Network import KerasNetwork as KN


class When(KN.KerasNetwork):
    """ classification neural network """

    def __init__(self, test):
        """ """

        # load network
        self.test = test
        KN.KerasNetwork.__init__(self, '{} when'.format(test.name),
                                 test.when_network_path) 
        self.labels = ['wait', 'action', 'end']
        self.load_network()


    ### NETWORK ###

    def create_network(self, n_hidden=64, n_layers=3):
        """ create ANN reward nertwork with keras """
        # first layer
        network = Sequential()
        network.add(Dense(n_hidden, activation='relu',
                          input_shape=self.test.mid_size))
        # hidden layers
        for _ in range(n_layers - 1):
            network.add(Dense(n_hidden, activation='relu'))
        # last layer
        network.add(Dense(len(self.labels), activation='softmax'))
        network.compile(loss='categorical_crossentropy', optimizer='adam')
        self.network = network


    ### TRAIN - ONLINE ###  

    def train_network(self, data, labels, bs=2, ep=50, cutoff=1e-4):
        """ action = (button, x, y, time) other """
        # format input data
        print('\nTraining when')
        labels = [DT.new_label(self.labels.index(lab), len(self.labels))
                  for lab in labels if lab in self.labels]
        labels = np.array(labels)
        print(' - data shape: {}'.format(data.shape))
        print(' - labels shape: {}'.format(labels.shape))
        print(labels)
        # loop until under cutoff
        count, cost = 0, 100
        while cost > cutoff and count < 30:
            self.network.fit(data, labels, batch_size=bs, epochs=ep, verbose=0)
            cost = self.network.evaluate(data, labels, batch_size=bs, verbose=0)
            print(' - cost {}: {}'.format(count, cost))
            count += 1


