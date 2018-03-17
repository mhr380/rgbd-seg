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

import matplotlib.pyplot as plt


class ImageDataGenerator(object):
    def __init__(self):
        self.__bit16 = 65535
        self.__bit8 = 255

        self.dirname = None
        self.reset()

        self.out_height = 480
        self.out_width = 640


    def reset(self):
        self.images = []
        self.labels = []

    def flow_from_directory(self, dirname, height=480, width=640, batch_size=32):

        self.dirname = dirname
        self.out_height = height
        self.out_width = width

        out_list = []
        for state in ['train', 'test']:
            image_names = os.listdir(
                os.path.join(self.dirname, state, 'color')
                )

            num_samples = len(image_names)

            # make [[color_name, depth_name], label_name] list
            data_path_list = []
            for image_name in image_names:
                color_path = os.path.join(self.dirname, state, 'color', image_name)
                depth_path = os.path.join(self.dirname, state, 'depth', image_name)

                label_path = os.path.join(self.dirname, state, 'label', image_name)

                data_path_list.append(
                    ([color_path, depth_path], label_path))


            # random shuffle
            data_path_list = np.random.permutation(data_path_list)

            x_list = []
            y_list = []

            for (imagedepth_path, label_path) in data_path_list:
                image_path = imagedepth_path[0]
                depth_path = imagedepth_path[1]

                image = self.__load_image(image_path)
                depth = self.__load_depth(depth_path)

                label = self.__load_label(label_path)

                # make 4ch array (B, G, R, D)
                x = np.dstack((image, depth))
                x_list.append(x)
                y_list.append(label)

            out_list.append((x_list, y_list))

        return out_list

    def __load_depth(self, filename, depth_max=1.0):
        ''' loads 16bit depth image and
            returns depth as 1ch np.float32 array.

            @Args:
                filename: path to depth image
                depth_max: max value of depth
            @Returns
                depth: np.float32 array.
        '''

        depth_16bit = cv2.imread(filename, cv2.IMREAD_ANYDEPTH)
        if depth_16bit is None:
            print('Load Error: {}'.format(filename))
            sys.exit()

        depth = np.float32(depth_16bit) / float(self.__bit16) * depth_max
        depth = cv2.resize(depth, (self.out_width, self.out_height))
        print(depth.shape)
        return depth

    def __load_image(self, filename, image_max=1.0):

        image_8bit = cv2.imread(filename, 1)
        if image_8bit is None:
            print('Load Error: {}'.format(filename))
            sys.exit()

        image = np.float32(image_8bit) / float(self.__bit8) * image_max
        image = cv2.resize(image, (self.out_width, self.out_height))

        return image

    def __load_label(self, filename):
        label = cv2.imread(filename, 0)

        if label is None:
            print('Load Error: {}'.format(filename))
            sys.exit()

        label = cv2.resize(label, (self.out_width, self.out_height))
        return np.float32(label)


if __name__ == '__main__':

    d = ImageDataGenerator()
    d.flow_from_directory('data')
