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

import keras
from keras import backend as K
from modules.io import ImageDataGenerator

from net import models


def main():

    #
    # params
    #
    height = 240
    width = 320
    ch = 4  # (B, G, R, D)
    class_num = 14
    batch_num = 16

    #
    # load data
    #
    loader = ImageDataGenerator()

    (x_train, y_train), (x_test, y_test) = loader.flow_from_directory('data_tmp', height=240, width=320)
    # dnum, height, width, ch

    #
    # load model
    #
    input_shape = (height, width, ch)
    model = models.FCN_VGG16_32s(input_shape, class_num)

    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(),
        metrics=['accuracy']
        )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_num,
        verbose=True,
        validation_data=(x_test, y_test)
        )

    score = model.evaluate(x_test, y_test)

    print('test loss: {:.5f}'.format(score[0]))
    print('test acc:  {:.5f}'.format(score[1]))

    model.save('out.h5')


if __name__ == '__main__':
    main()
