import os
import numpy as np

from Library.General import DataThings as DT
from Library.General import Screen
from Library.TestAutomation import When
from Library.TestAutomation import What
from Library.TestAutomation import Where


class Test(object):
    """ """

    def __init__(self, base_path, test_name, auto_network, window, width, height):
        """ """

        # location
        self.base_path = os.path.join(base_path, test_name)
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.when_network_path = os.path.join(self.base_path, 'when')
        self.what_network_path = os.path.join(self.base_path, 'what')
        self.where_network_path = os.path.join(self.base_path, 'where')

        # network
        self.auto_network = auto_network
        self.mid_size = (256,)

        # screen
        self.window = window
        self.width, self.height = width, height

        # TA params
        self.name = test_name
        self.when = When.When(self)
        self.what = What.What(self)
        self.where = Where.Where(self)

        # text params
        self.text_file = os.path.join(self.base_path, 'default_strings.txt')
        self.text_index = 0
        if os.path.exists(self.text_file):
            self.text_list = DT.read_file(self.text_file)


    ### TEXT ###

    def next_string(self):
        """ """
        text = self.text_list[self.text_index]
        self.text_index += 1
        if self.text_index == len(self.text_list):
            self.text_index = 0
        return text

    def record_text(self, n_times=1):
        """ """
        print('Record text params for test...')
        count, texts = 0, []
        while count < n_times:
            count += 1
            texts.append(input(' - param {}/{}:'.format(count, n_times)))
        with open(self.text_file, 'w') as file:
            file.write('\n'.join(texts))
        

    ### WINDOW ###

    def get_window(self, x=1024, y=1024):
        """ """
        return Screen.get_data_resized(x, y)

    def get_state(self):
        """ gets current screen state from screen """
        ds = np.reshape(self.get_window(), (-1, 1024, 1024, 3))
        auto_mid = np.reshape(self.auto_network.get_flat(ds), (-1, self.mid_size[0]))
        return auto_mid

    def screencap_window(self, name='main_window'):
        """ """
        save_path = os.path.join(self.base_path, '{}.png'.format(name))
        Screen.save_image(self.get_window(), save_path)


    ### NETWORK ###

    def save_networks(self):
        """ """
        self.when.save_network()
        self.what.save_network()
        self.where.save_network()


