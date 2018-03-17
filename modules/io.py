# coding: utf-8

"""
@file       io.py
@brief      modules for images, labels, and depth io.
@author		mhr380
@requirements opencv-python, numpy
"""

import sys
import os

import cv2
import numpy as np


def load_depth(filename, depth_max=1.0):
    ''' loads 16bit depth image and
        returns depth as 1ch np.float32 array.

        @Args:
            filename: path to depth image
            depth_max: max value of depth
        @Returns
            depth: np.float32 array.
    '''

    bit16 = 65535

    try:
        depth_16bit = cv2.imread(filename, -1)
    except:
        print('Load Error: {}'.format(filename))
        sys.exit()

    depth = np.float32(depth_16bit) / float(bit16) * depth_max

    return depth


def load_image(filename, image_max=1.0):

    bit8 = 255

    try:
        image_8bit = cv2.imread(filename, 1)
    except:
        print('Load Error: {}'.format(filename))
        sys.exit()

    image = np.float32(image_8bit) / float(bit8) * image_max

    return image


def load_label(filename):

    try:
        label = cv2.imread(filename, 1)
    except:
        print('Load Error: {}'.format(filename))
        sys.exit()

    return label

