import os
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from PIL import Image

from Library.Network import NetworkAPI as NETS


### TEXT FILE ###

def read_file(filename):
    """ return lines from text file """
    with open(filename, 'r') as file:
        return file.read().split('\n')


### IMAGE FILE ###

def load_image(file):
    """ returns normalized data for image file """
    return np.array(Image.open(file))[:, :, :3] / 255

def load_images(fpath, n_files=0):
    """ returns portion of normalized data for images files """
    i = len(os.listdir(fpath)) if n_files == 0 else n_files
    return np.array([load_image(join(fpath, f)) for f in os.listdir(fpath)[:i]])


### NETWORK FILE ###

def load_keras_network(filename):
    """ load keras neural network """
    return load_model(filename)

def load_auto(path, auto_name):
    """ load tensorflow neural network """
    return NETS.load_auto(path, auto_name)


### IAMGE ###

def pad_me(data, pad1, pad2):
    """ pad array of images at edges """
    return np.pad(data, ((0, 0), (pad1, pad1), (pad2, pad2), (0, 0)),
                  mode='constant', constant_values=0)

def subdata(data, height, width):
    """ get subimage of size given image data """
    i = np.random.randint(data.shape[1] - height)
    j = np.random.randint(data.shape[2] - width)
    return data[:, i:i + height, j:j + width, :]

def subdata_xy(data, height, width, x, y):
    """ get subimage at coordinates given image data """
    return data[:, x-height//2:x+height//2, y-width//2:y+width//2, :]


### LABELS ###

def new_label(idx, n_classes):
    """ new one-hot label given index and size """
    label = np.zeros(n_classes)
    label[idx] = 1
    return label

def to_one_hot(labels, n_classes=0):
    """ new set of one-hot labels given indexed labels """
    label_set = list(sorted(set(labels)))
    n_classes = len(label_set) if n_classes == 0 else n_classes
    one_hot = [new_label(label_set.index(lab), n_classes) for lab in labels]
    return np.array(one_hot)


### PLOT ###

def plot_data_multiple(data, labels=None, n_x=3, n_y=6, figure_size=(16, 8),
                       save_path=False):
    """ plot data using matplotlib """
    fig = plt.figure(figsize=figure_size)
    # subplots
    for i in range(min(n_x * n_y, len(data))):
        ax = fig.add_subplot(n_x, n_y, i + 1)
        ax.imshow(data[i])
        ax.set_aspect('equal')
        ax.margins(x=0, y=0)
        if labels is not None:
            ax.set_title(labels[i])
        ax.axis('off')
    # other
    cax = fig.add_axes([0.05, 0.05, 0.95, 0.95])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    fig.set_tight_layout(True)
    _ = [fig.savefig(save_path) if save_path else fig.show() for i in range(1)]
    fig.clf()


