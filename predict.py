# coding: utf-8

"""
@file       train.py
@brief      excecutes training
@author		mhr380
@requirements   keras
                tensorflow==1.4.0
                opencv-python
                numpy
"""

import sys
import os

import cv2
import numpy  as np
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from  keras.models import load_model

from modules.io import ImageDataGenerator

from net import models


def colorize(onehotarray):

    class_num = 14
    color = [0] * class_num

    color[0] = [0,0,0]
    color[1] = [128,0,0]
    color[2] = [192,192,128]
    color[3] = [255,69,0]
    color[4] = [128,64,128]
    color[5] = [60,40,222]
    color[6] = [128,128,0]
    color[7] = [192,128,128]
    color[8] = [64,64,128]
    color[9] = [64,0,128]
    color[10] = [64,64,0]
    color[11] = [0,128,192]
    color[12] = [128,128,128]
    color[13] = [0, 0, 128]

    height, width = onehotarray.shape[:2]
    outarray = np.zeros((height, width, 3), np.uint8)

    for h in range(height):
        for w in range(width):
            vec = onehotarray[h, w, :]
            idx = np.argmax(vec)
            outarray[h, w, :] = color[idx]

    return outarray


if __name__ == '__main__':

    # params
    #
    height = 480
    width = 640
    ch = 4  # (B, G, R, D)
    class_num = 14
    batch_num = 16

    #
    # load data
    #
    loader = ImageDataGenerator()

    (x_train, y_train), (x_test, y_test) = loader.flow_from_directory('data_tmp', height=height, width=width)
    # dnum, height, width, ch

    #
    # load model
    #
    input_shape = (height, width, ch)
    model = load_model('out.h5')


    for i in range(3):
        i = x_test[i]
        i = i[None, :, :, :]
        y = model.predict(i)[0, ...]
        print(y.shape)

        colorized = colorize(y)
        cv2.imshow("win", colorized)
        cv2.waitKey(0)
