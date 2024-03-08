import os
from os.path import join


# top level
base_path = os.path.dirname(os.path.realpath(__file__))
data_path = join(base_path, 'data')

# data types
images_path = join(data_path, 'images')
network_path = join(data_path, 'networks')
tests_path = join(data_path, 'tests')

# images
auto_imgs_path = join(images_path, 'auto_images')


