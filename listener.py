import os
import time
import numpy as np
from PIL import ImageGrab
from threading import Thread

import paths
from Library.General import Screen


### PARAMS ###

# screen listener
screen_data = []
screen_listener_on = False
action_listener_on = False
start_time = 0


### ACTION ###

def is_valid_action(action, w=1920, h=1080):
    """ """
    return action[1] > 0 and action[1] < w and action[2] > 0 and action[2] < h

def done_taking_actions(actions):
    """ """
    #return not int(actions[-1][1]) > 1920
    return not int(actions[-1][1]) < 0


### LISTENER ###

def generate_auto_data(sleep_time=0.5):
    """ """
    done = False
    count = 0
    while not done:
        path = os.path.join(paths.auto_imgs_path, 'data_{}.png'.format(count))
        Screen.screenshot(path)
        time.sleep(sleep_time)
        count+= 1
        if count % 5 == 0:
            print(count)

def listen_to_actions(tp, sleep_time_screenshot=0.5, print_me=False):
    """ action_data = (button, x, y, time) """
    # reset listener params
    global screen_data, screen_listener_on, action_listener_on, start_time
    screen_data = []
    screen_listener_on = True
    action_listener_on = True
    start_time = time.time()
    # start thread to save screen images
    t = Thread(target=listen_to_screen, args=(sleep_time_screenshot,))
    t.start()
    # listen to actions until done taking actions
    print('Listening to screen and actions...')
    actions = []
    while action_listener_on:
        actions.append(list(Screen.get_click()) + [time.time() - start_time])
        screen_listener_on = done_taking_actions(actions)
        print(' - action: {}'.format(actions[-1][0]))
    sa, ss, od, ol = clean_data(actions, screen_data, tp)
    return sa, ss, od, ol

def listen_to_screen(sleep_time):
    """ screen_data = (full_image, time) """
    global screen_data, screen_listener_on, action_listener_on, start_time
    while screen_listener_on:
        screen_data.append((ImageGrab.grab(), time.time() - start_time))
        time.sleep(sleep_time)
    print('Finished listening to screen...')
    action_listener_on = False

def clean_data(actions, screen_data, tp):
    """ """
    actions = clean_actions(actions)
    sa, ss, od, ol = time_stuff(screen_data, actions, tp)
    return sa, ss, od, ol


### ACTION ###

def clean_actions(actions, click_cutoff=0.5):
    """ action_data = (button, x, y, time) """
    print('Cleaning actions...')
    # loop over all actions and convert to click or drag
    k, new_actions = 0, []
    while k < len(actions) - 1:
        if is_valid_action(actions[k]):
            new_actions, k = check_click(actions, new_actions, k)
            new_actions = check_double_click(new_actions)             
        k += 1
    return new_actions

def new_name(text1, text2):
    """ action_data = (button, x, y, time) """
    return text1.split('_')[0] + '_' + text2

def check_click(actions, new_actions, k, click_cutoff=0.5):
    """ action_data = (button, x, y, time) """
    # is press and release, then distinguish click vs drag
    if 'pre' in actions[k][0] and 'rel' in actions[k + 1][0]:
        a1, a2 = actions[k], actions[k + 1]
        click_and_drag = a2[-1] - a1[-1] > click_cutoff
        new_action = [new_name(a1[0], 'drag' if click_and_drag else 'click'),
                      a1[1], a1[2], a2[1], a2[2], a2[-1]]
        new_actions.append(new_action)
        k += 1
    return new_actions, k

def check_double_click(actions, click_cutoff=0.5):
    """ action_data = (button, x, y, time) """
    # remove last two actions and replace with double if checks True
    if len(actions) > 1:
        is_click = 'click' in actions[-1][0]
        same_click_name = actions[-1][0] ==  actions[-2][0]
        fast_clicks = actions[-1][-1] - actions[-2][-1] < click_cutoff
        if is_click and same_click_name and fast_clicks:
            del actions[-1]
            actions[-1][0] = new_name(actions[-1][0], 'double')
    return actions


### TIME ###

def time_stuff(screen_data, actions, tp, cutoff=0.01, reaction=0.4, s=1024):
    """ """
    print('Matching actions with screenshots...\n')
    # time values 
    screen_times = [sd[-1] + reaction for sd in screen_data]
    action_times = [a[-1] for a in actions]
    # match unique actions to unique screenshots
    screens_for_actions, sub_actions = get_action_screens(screen_data, screen_times, actions, tp)
    other_data, other_labels = things(screen_data, screen_times, actions, tp)
    # stats
    print(' - total screens: {}'.format(len(screen_data)))
    print(' - screens saved: {}'.format(len(other_labels)))
    print(' - total actions: {}'.format(len(actions)))
    print(' - actions saved: {}'.format(len(sub_actions)))
    return sub_actions, screens_for_actions, other_data, other_labels

def find_screen_action_idxs(screen_times, actions):
    """ action_data = (button, x, y, time) """
    # find screens where actions happen
    screen_idxs, action_idxs = [], []
    for idx_a, action in enumerate(actions):
        idx_s = [i for i, t in enumerate(screen_times) if t < action[-1]][-1]
        if idx_s not in screen_idxs:
            screen_idxs.append(idx_s)
            action_idxs.append(idx_a)
    return screen_idxs, action_idxs

def get_action_screens(screen_data, screen_times, actions, tp, save_me=True, s=1024):
    """ action_data = (button, x, y, time) """
    # get sub screens and sub actions
    screen_idxs, action_idxs = find_screen_action_idxs(screen_times, actions)
    sub_screens = np.array([Screen.resize_image(screen_data[i][0], s, s)
                            for i in screen_idxs])
    sub_actions = [actions[i] for i in action_idxs]
    # save images
    if 0:
        print('Saving action images...')
        for i, image_data in enumerate(sub_screens):
            params = [screen_idxs[i]] + list(sub_actions[i][:3])
            path = os.path.join(tp, 'test_{}_{}_{}_{}.png'.format(*params))
            Screen.save_image(np.array(image_data), path)
    return sub_screens, sub_actions

def things(screen_data, screen_times, actions, tp, save_me=True):
    """ """
    # get screen indexs from actions and load data
    screen_idxs, _ = find_screen_action_idxs(screen_times, actions)
    idxs = screen_idxs.copy()
    idxs += [len(screen_data) - 1]
    old_data = [np.array(screen_data[i][0]) for i in idxs]
    # add other data if unique enough
    for i, sd in enumerate(screen_data):
        if i not in idxs and unique_enough(sd[0], old_data):
            idxs += [i]
            old_data.append(np.array(screen_data[i][0]))
    # get data and labels
    data = np.array([Screen.resize_image(screen_data[i][0], 1024, 1024)
                     for i in idxs])
    labels = ['action' if i in screen_idxs else 'wait' for i in idxs]
    labels[len(screen_idxs)] = 'end'
    # save images
    if 0:
        print('Saving other images...')
        for i, image_data in enumerate(data):
            path = os.path.join(tp, 'test_{}_{}.png'.format(idxs[i], labels[i]))
            Screen.save_image(np.array(image_data), path)
    return data, labels

def unique_enough(img1, datas, cutoff=0.05):
    """ """
    diffs = [difference(np.array(img1), d) for d in datas]
    avgs = [d > cutoff for d in diffs]
    #print(diffs)
    return all(avgs)

def difference(array1, array2):
    """ """
    return np.mean(array1 - array2) / 255


