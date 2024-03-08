import os
import numpy as np
from PIL import Image, ImageGrab
from pynput import mouse, keyboard
from pynput.mouse import Button as MB

import win32api
import win32con
import pyautogui


### PARAMS ###

# global listeners
mbutton, mx, my, mpressed = 0, 0, 0, 0
mouseAPI = mouse.Controller()
mouseDict = {MB.left: 'left', MB.right: 'right', MB.middle: 'middle'}
mouseDictRev = {v: k for k, v in mouseDict.items()}

keyAPI = keyboard.Controller()


### SCREEN ###

def get_data():
    """ returns a normalized array of screen image pixels """
    return np.array(ImageGrab.grab()) / 255

def get_data_resized(width, height):
    """ returns resized screen pixel data """
    img = ImageGrab.grab()
    return np.array(img.resize((height, width), Image.ANTIALIAS)) / 255

def resize_image(img, width, height):
    """ returns resized image """
    return np.array(img.resize((height, width), Image.ANTIALIAS)) / 255

def screenshot(save_path):
    """ """
    save_image(get_data(), save_path)

def save_image(data, save_path):
    """ saves data as image to path, un-normalizing if necessary """
    data = data * 255 if data.max() <= 1.0 else data
    image = Image.fromarray(data.astype('uint8'))
    image.save(save_path)


### MOUSE ACTION ###

def move_to(x, y):
    """ move mouse to coordinates """
    mouseAPI.position = (x, y)

def click(x, y, button, n_click=1):
    """ move mouse to coordinates and mouse click """
    move_to(x, y)
    mouseAPI.click(mouseDictRev[button], n_click)

def press(x, y, button):
    """ move mouse to coordinates and mouse press """
    move_to(x, y)
    mouseAPI.press(mouseDictRev[button])

def release(x, y, button):
    """ move mouse to coordinates and mouse release """
    move_to(x, y)
    mouseAPI.release(mouseDictRev[button])

def click_drag(x1, y1, x2, y2):
    """ """
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x1, y1, 0, 0)
    pyautogui.moveTo(x2, y2, duration=1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x2, y2, 0, 0)


### KEY ACTION ###

def send_keys(text):
    """ type keys to keyboard given text """
    keyAPI.type(text)

def key_enter():
    """ """
    keyAPI.press(keyboard.Key.enter)

### MOUSE LISTENER ###

def on_click(x, y, button, pressed):
    """ pynput mouse action listener on click """
    global mbutton, mx, my, mpressed
    mbutton = '{}_{}'.format(mouseDict[button], 'pre' if pressed else 'rel')
    mx, my, mpressed = int(x), int(y), pressed
    return False

def get_click(message=None):
    """ start mouse listener and return location and type of next click """
    global mbutton, mx, my, mpressed
    mbutton, mx, my, mpressed = 0, 0, 0, 0
    if message:
        print(message)
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    return mbutton, mx, my


