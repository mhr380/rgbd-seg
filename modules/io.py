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

from keras.preprocessing.image import load_img, img_to_array


class ImageDataGenerator(object):
    def __init__(self):
        self.__bit16 = 65535
        self.__bit8 = 65535

        self.dirname = None
        self.reset()

    def reset(self):
        self.images = []
        self.labels = []

    def flow_from_directory(self, dirname, batch_size=32):

        self.dirname = dirname

        out_list = []
        for state in ['train', 'test']:
            image_names = os.listdir(
                os.path.join(self.dirname, state, 'color')
                )

            num_amples = len(image_names)

            # make [[color_name, depth_name], label_name] list
            data_path_list = []
            for image_name in image_names:
                color_path = os.path.join(self.dirname, 'train', 'color')
                depth_path = os.path.join(self.dirname, 'train', 'depth')

                label_path = os.path.join(self.dirname, 'train', 'label')

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
                depth = self.__load_image(depth_path)

                label = self.__load_image(label_path)

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

        try:
            depth_16bit = cv2.imread(filename, -1)
        except:
            print('Load Error: {}'.format(filename))
            sys.exit()

        depth = np.float32(depth_16bit) / float(self.__bit16) * depth_max
        return depth

    def __load_image(self, filename, image_max=1.0):

        try:
            image_8bit = cv2.imread(filename, 1)
        except:
            print('Load Error: {}'.format(filename))
            sys.exit()

        image = np.float32(image_8bit) / float(self.__bit8) * image_max

        return image

    def __load_label(self, filename):

        try:
            label = cv2.imread(filename, 1)
        except:
            print('Load Error: {}'.format(filename))
            sys.exit()

        return np.float32(label)


if __name__ == '__main__':

    d = ImageDataGenerator()
    d.flow_from_directory(self, 'data', batch_size=32)
