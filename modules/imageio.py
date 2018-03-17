# coding: utf-8

"""
@file		imageio.py
@brief      modules for depth io.
            - loads uint16 depth image (0 - 65535)
                and converts to float32 image (0[m]-8[m]).
            - writes float32 dpeth image 
@author		Hajime Mihara

@requirements cv2
"""

import sys
import os

import numpy as np
import h5py



def depthio():