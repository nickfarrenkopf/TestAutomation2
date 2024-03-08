import os
import time
import itertools
import numpy as np
import tensorflow as tf


class Network(object):
    """ base network """

    def __init__(self, save_path, name):
        """ params - name, path, session, and graph """
        self.name = name
        self.save_path = os.path.join(save_path, name)
        self.graph = tf.Graph()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True),
                               graph=self.graph)
        self.load_main()


    ### FILE ###

    def load_main(self):
        """ load main parts of network """
        with self.sess.as_default():
            with self.graph.as_default():
                saver = tf.train.import_meta_graph(self.save_path + '.meta')
                saver.restore(self.sess, self.save_path)
                get_tensor = self.graph.get_tensor_by_name
                self.inputs = get_tensor('{}/inputs:0'.format(self.name))
                self.outputs = get_tensor('{}/outputs:0'.format(self.name))
                self.alpha = get_tensor('{}/alpha:0'.format(self.name))
                self.cost = get_tensor('{}/cost:0'.format(self.name))
                self.train = tf.get_collection('{}_train'.format(self.name))[0]

    def save_network(self, step=0):
        """ saves network to save path """
        print('Saving network {}'.format(self.name))
        with self.sess.as_default():
            with self.graph.as_default():
                saver = tf.train.Saver()
                if step == 0:
                    saver.save(self.sess, self.save_path)
                else:
                    saver.save(self.sess, self.save_path, global_step=step)



class NetworkAuto(Network):
    """ Convolutional Autoencoder Network """

    def __init__(self, save_path, name):
        """ initializes network and load extras """
        start = time.time()
        Network.__init__(self, save_path, name)
        self.flat_size = int(np.sqrt(int(name.split('_')[-1])))
        self.mid_size = (self.flat_size, self.flat_size)
        self.load_extras()


    ### FILE ###

    def load_extras(self):
        """ load additional network layers """
        with self.sess.as_default():
            with self.graph.as_default():
                get_tensor = self.graph.get_tensor_by_name
                self.flat = get_tensor('{}/flat:0'.format(self.name))

    def batch_me(self, data, func, alpha=0, batch=4):
        """ FIX ME? """
        datas = []
        for i in range(data.shape[0] // batch + 1):
            subdata = data[batch * i : batch * (i + 1)]
            if subdata.shape[0] == 0:
                    break
            while subdata.shape[0] < batch:
                subdata = np.array(list(subdata) + list(subdata[:1]))
            feed = {self.inputs: subdata, self.alpha : alpha}
            res = self.sess.run(func, feed)
            if res is None:
                res = [0]
            if isinstance(res, np.float32):
                res = np.array([res])
            datas.append(res)
        r = np.array(list(itertools.chain.from_iterable(datas)))
        return r[:data.shape[0]]
            

    ### TENSORS ###

    def get_flat(self, input_data):
        """ return middle flat layer of network given feed """
        return self.batch_me(input_data, self.flat)

    def get_outputs(self, input_data):
        """ return outputs of network given feed """
        return self.batch_me(input_data, self.outputs)

    def get_cost(self, input_data):
        """ return cost of network given feed """
        return np.mean(self.batch_me(input_data, self.cost))

    def train_network(self, input_data, alpha):
        """ train network given feed """
        _ = self.batch_me(input_data, self.train, alpha=alpha)


