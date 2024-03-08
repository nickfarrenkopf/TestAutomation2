import os
from keras.models import load_model


class KerasNetwork(object):
    """ """

    def __init__(self, name, network_path):
        """ """
        self.name = name
        self.network_path = network_path


    ### FILE ###

    def load_network(self):
        """ load keras network if exists, create otherwise """
        if not os.path.exists(self.network_path):
            self.create_network()
            self.save_network()
        else:
            self.network = load_model(self.network_path)
        
    def save_network(self):
        """ save keras network """
        self.network.save(self.network_path)
        print('{} network saved to {}'.format(self.name, self.network_path))


    ### HELPER ###

    def print_metrics(self, data, labels):
        """ print metrics of network """
        metrics = self.network.evaluate(data, labels, verbose=0)
        print('Metrics: {}'.format(metrics))

    def predict(self, data):
        """ get prediction given input data """
        return self.network.predict(data, verbose=0)[0]


