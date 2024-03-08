import os
import time
import numpy as np
import pyautogui
from win32api import GetSystemMetrics

import paths
import listener

from Library.General import Screen
from Library.General import DataThings as DT
from Library.TestAutomation import Test


### TRAIN  ###

def run_trainer(save_me=True):
    """ """
    params = listener.listen_to_actions(test.base_path)
    print('\nAction list')
    for action in params[0]:
        print(' - {}'.format(action))
    # train networks
    train_networks(*params)
    # save networks
    if save_me:
        test.save_networks()
        sums = [1 for a in params[0] if 'left_double' in a[0]]
        #test.record_text(n_times=sum(sums))

def train_networks(actions, action_data, other_data, other_labels):
    """ """
    # flat network
    action_data = auto_network.get_flat(action_data)
    other_data = auto_network.get_flat(other_data)
    # test text params
    test.when.train_network(other_data, other_labels)
    test.what.train_network(action_data, actions)
    test.where.train_network(action_data, actions, base_width, base_height)
    

### PREDICT ###

def pred_when(data):
    """ """
    when_pred = test.when.predict(data)
    return test.when.labels[list(when_pred).index(when_pred.max())]

def pred_what(data):
    """ """
    what_pred = test.what.predict(data)
    return test.what.labels[list(what_pred).index(what_pred.max())]

def pred_where(data):
    """ """
    where_pred = test.where.predict(data)
    x1 = int(where_pred[0] * base_width)
    y1 = int(where_pred[1] * base_height)
    x2 = int(where_pred[2] * base_width)
    y2 = int(where_pred[3] * base_height)
    return x1, y1, x2, y2


### TEST ###

def test_online_when(sleep_time=0.2):
    """ """
    print('Testing when...')
    done = False
    while not done:
        print(' - {}'.format(pred_when(test.get_state())))
        time.sleep(sleep_time)

def test_online_adhoc():
    """ """
    print('Testing test...')
    input_text = ''
    while input_text != 'q':
        ds = test.get_state()
        print('When: {}'.format(pred_when(ds)))
        print('\nWhat: {}'.format(pred_what(ds)))
        print('\nWhere: ({}, {}) ({}, {})'.format(*pred_where(ds)))
        input_text = input('Press enter to take action... (q to quit)')


### EXECUTE ###

def execute_test(max_steps=10):
    """ """
    steps = 0
    while execute_turn() and steps < max_steps:
        print('Checking screen...')
        steps += 1
    print('Done')

def execute_turn(action_sleep_time=3, wait_sleep_time=3):
    """ """
    # when
    ds = test.get_state()
    when_label = pred_when(ds)
    # action
    if when_label == 'action':
        what_label = pred_what(ds)
        x1, y1, x2, y2 = pred_where(ds)
        print(' - action {} {} {} {} {}'.format(what_label, x1, y1, x2, y2))
        execute_action(what_label, x1, y1, x2, y2)
        time.sleep(action_sleep_time)
    # wait
    if when_label == 'wait':
        print(' - waiting {}...'.format(wait_sleep_time))
        time.sleep(wait_sleep_time)
    # end
    if when_label == 'end':
        print(' - test over')
        return False
    return True

def execute_action(button_text, x1, y1, x2, y2):
    """ """
    button_choice, action_type = button_text.split('_')
    # click and drag
    if action_type == 'drag':
        Screen.click_drag(x1, y1, x2, y2)
    # left/right mouse click
    elif action_type == 'click':
        Screen.click(x1, y1, button_choice)
    # middle mouse click - type text
    elif action_type == 'double':
        Screen.click(x1, y1, button_choice)
        text = test.next_string()
        enter_count = 0
        while text[-1] == 'E':
            text = text[:-1]
            enter_count += 1
        Screen.send_keys(text)
        time.sleep(0.2)
        for i in range(enter_count):
            print('Enter')
            pyautogui.press('enter')
            time.sleep(0.2)
        time.sleep(2)
    else:
        print('Failed')


### PARAMS ###

# screen dimensions
base_height = GetSystemMetrics(0)
base_width = GetSystemMetrics(1)
base_window = ((0, 0), (base_height, base_width))


### PROGRAM ###

# LOAD AUTO
if 1:
    name = 'AUTO_testing_1024_1024_8_256'
    name = 'AUTO_test_auto_1024_1024_8_256'
    auto_network = DT.load_auto(paths.network_path, name)


# LOAD TEST
if 1:
    test_name = 'Test7'
    test = Test.Test(paths.tests_path, test_name, auto_network, base_window,
                     base_width, base_height)


# TRAIN TEST
if 0:
    run_trainer(True)


# TEST TEST
if 1:
    execute_test(max_steps=7)


