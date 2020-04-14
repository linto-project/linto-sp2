"""A vanilla 3D resnet implementation.
Based on Raghavendra Kotikalapudi's 2D implementation
keras-resnet (See https://github.com/raghakot/keras-resnet.)
"""
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    concatenate
)
from keras.layers.convolutional import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.utils import np_utils
import numpy as np
import glob
import random
import cv2, os
from tools.utils import noisy
from os.path import isfile

kernel_w = 224
kernel_h = 224
img_w = 320 # 171
img_h = 240 # 128

doFlip = True
doScale= True
doTrans= True
doNoisy= True


def process_batch_rgb(lines, img_path, train=True):
    num = len(lines)
    batch = np.zeros((num, 16, kernel_w, kernel_h, 3), dtype='float32')
    labels = np.zeros(num, dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol) - 1
        # imgs = os.listdir(img_path+path)
        # imgs = glob.glob(img_path + path + '/*.jpg')
        # if imgs == []:
        #     imgs = glob.glob(img_path + path + '/*.png')
        # imgs.sort(key=str.lower)

        if not os.path.isfile(path + '/{:06d}'.format(symbol + 16 ) + '.jpg'):
            continue
        if train:
            crop_x = random.randint(0, 10)
            crop_y = random.randint(0, 10)
            tran_x = random.randint(0, 10)
            tran_y = random.randint(0, 10)
            is_flip = random.randint(0, 1)

            sc = random.uniform(0.2, 1)

            for j in range(16):
                img_file = path + '/{:06d}'.format(symbol + j) + '.jpg'
                image = cv2.imread(img_file)
                # Re-scale
                if doScale:
                    hh, ww = image.shape[0:2]
                    hh = int(hh * sc)
                    ww = int(ww * sc)
                    image = cv2.resize(image, (ww, hh))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (img_w, img_h))

                if is_flip == 1 and doFlip:
                    image = cv2.flip(image, 1)

                if doTrans:
                    hh, ww = image.shape[0:2]
                    M = np.float32([[1, 0, tran_x], [0, 1, tran_y]])
                    image = cv2.warpAffine(image, M, (ww, hh))

                if doNoisy:
                    noisy(image, 'gauss')

                batch[i][j][:][:][:] = image[crop_x:crop_x + kernel_w, crop_y:crop_y + kernel_h, :]
            labels[i] = label
        else:
            for j in range(16):
                img_file = path + '/{:06d}'.format(symbol + j) + '.jpg'
                image = cv2.imread(img_file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (img_w, img_h))
                h_start = (img_h - kernel_h) // 2
                h_end = h_start + kernel_h
                w_start = (img_w - kernel_w) // 2
                w_end = w_start + kernel_w
                batch[i][j][:][:][:] = image[h_start:h_end, w_start:w_end, :]
            labels[i] = label
    return batch, labels


def preprocess(inputs):
    inputs[..., 0] -= 99.9
    inputs[..., 1] -= 92.1
    inputs[..., 2] -= 82.6
    inputs[..., 0] /= 65.8
    inputs[..., 1] /= 62.3
    inputs[..., 2] /= 60.3
    # inputs /=255.
    # inputs -= 0.5
    # inputs *=2.
    return inputs


def generator_train_batch(train_txt, batch_size, num_classes, img_path):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            x_train, x_labels = process_batch_rgb(new_line[a:b], img_path, train=True)
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            x = np.transpose(x, (0, 2, 3, 1, 4))
            yield x, y


def generator_val_batch(val_txt, batch_size, num_classes, img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test, y_labels = process_batch_rgb(new_line[a:b], img_path, train=False)
            x = preprocess(y_test)
            x = np.transpose(x, (0, 2, 3, 1, 4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def generator_test_batch(val_txt, batch_size, num_classes, img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        # random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test, y_labels = process_batch_rgb(new_line[a:b], img_path, train=False)
            x = preprocess(y_test)
            x = np.transpose(x, (0, 2, 3, 1, 4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


#  -----------------------------------------------------------------
# Siamese
#  -----------------------------------------------------------------


def siamese_process_batch(lines, img_path, train=True):
    num = len(lines)
    batch = np.zeros((num, 16, kernel_w, kernel_h, 3), dtype='float32')
    btcof = np.zeros((num,16,kernel_w, kernel_h,3),dtype='float32')
    labels = np.zeros(num, dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol) - 1
        # imgs = os.listdir(img_path+path)
        # imgs = glob.glob(img_path + path + '/*.jpg')
        # if imgs == []:
        #     imgs = glob.glob(img_path + path + '/*.png')
        # imgs.sort(key=str.lower)
        if not os.path.isfile(path + '/{:06d}'.format(symbol + 16 ) + '.jpg'):
            continue

        file_path = path + '/{:06d}'.format(symbol)
        flow_ext = ''
        use_npy = True
        if isfile(file_path + '.npy'):
            flow_ext = '.npy'
        elif isfile(file_path + '.npz'):
            flow_ext = '.npz'
        else:
            use_npy = False
            flow_ext = '_flow.jpg'

        if train:
            # crop_x = random.randint(0, 15)
            # crop_y = random.randint(0, 58)
            # is_flip = random.randint(0, 1)
            crop_x = random.randint(0, 10)
            crop_y = random.randint(0, 10)
            tran_x = random.randint(0, 10)
            tran_y = random.randint(0, 10)
            is_flip = random.randint(0, 1)

            sc = random.uniform(0.2, 1)

            for j in range(16):
                file_path = path + '/{:06d}'.format(symbol + j)
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(file_path + '.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = cv2.resize(image, (img_w, img_h))

                if not os.path.isfile(file_path + flow_ext):
                    print(file_path + flow_ext)
                if use_npy:
                    im_of = np.load(file_path + flow_ext)
                else:
                    im_of = cv2.imread(file_path + flow_ext)
                if im_of.shape[2] == 2:
                    mag, ang = cv2.cartToPolar(im_of[..., 0], im_of[..., 1])
                    im_of = np.dstack((im_of, mag))
                # im_of *= 16
                im_of = cv2.normalize(im_of, None, -1, 1, cv2.NORM_MINMAX)
                # im_of = cv2.resize(im_of, (img_w, img_h))

                # image = cv2.resize(image, (kernel_w, kernel_h))
                # im_of = cv2.resize(im_of, (kernel_w, kernel_h))
                # batch[i][j][:][:][:] = image
                # btcof[i][j][:][:][:] = im_of
                # continue

                # Re-scale
                if doScale:
                    hh, ww = image.shape[0:2]
                    hh = int(hh * sc)
                    ww = int(ww * sc)
                    image = cv2.resize(image, (ww, hh))
                    im_of = cv2.resize(im_of, (ww, hh))

                if is_flip == 1 and doFlip:
                    image = cv2.flip(image, 1)
                    im_of = cv2.flip(im_of, 1)

                if doTrans:
                    hh, ww = image.shape[0:2]
                    M = np.float32([[1, 0, tran_x], [0, 1, tran_y]])
                    image = cv2.warpAffine(image, M, (ww, hh))
                    im_of = cv2.warpAffine(im_of, M, (ww, hh))

                if doNoisy:
                    noisy(image, 'gauss')
                    noisy(im_of, 'gauss')

                image = cv2.resize(image[crop_x:, crop_y:, :], (kernel_w, kernel_h))
                im_of = cv2.resize(im_of[crop_x:, crop_y:, :], (kernel_w, kernel_h))
                batch[i][j][:][:][:] = image
                btcof[i][j][:][:][:] = im_of
            labels[i] = label
        else:
            for j in range(16):
                file_path = path + '/{:06d}'.format(symbol + j)
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(file_path + '.jpg')
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if use_npy:
                    im_of = np.load(file_path + flow_ext)
                else:
                    im_of = cv2.imread(file_path + flow_ext)

                if im_of.shape[2] == 2:
                    mag, ang = cv2.cartToPolar(im_of[..., 0], im_of[..., 1])
                    im_of = np.dstack((im_of, mag))
                im_of = cv2.normalize(im_of, None, -1, 1, cv2.NORM_MINMAX)

                image = cv2.resize(image, (kernel_w, kernel_h))
                im_of = cv2.resize(im_of, (kernel_w, kernel_h))
                batch[i][j][:][:][:] = image
                btcof[i][j][:][:][:] = im_of
                continue

                image = cv2.resize(image, (img_w, img_h))
                im_of = cv2.resize(im_of, (img_w, img_h))
                h_start = (img_h - kernel_h) // 2
                h_end = h_start + kernel_h
                w_start = (img_w - kernel_w) // 2
                w_end = w_start + kernel_w
                batch[i][j][:][:][:] = image[h_start:h_end, w_start:w_end, :]
                btcof[i][j][:][:][:] = im_of[h_start:h_end, w_start:w_end, :]
            labels[i] = label
    return batch, btcof, labels


def siamese_generator_train_batch(train_txt, batch_size, num_classes, img_path):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            x_train, x2_train, x_labels = siamese_process_batch(new_line[a:b],img_path,train=True)
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            x = np.transpose(x, (0, 2, 3, 1, 4))
            x2= np.transpose(x2_train, (0,2,3,1,4))
            yield [x, x2], y


def siamese_generator_val_batch(val_txt, batch_size, num_classes, img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            # y_test, y_labels = process_batch(new_line[a:b], img_path, train=False)
            y_test, y2_test, y_labels = siamese_process_batch(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            x = np.transpose(x, (0, 2, 3, 1, 4))
            x2= np.transpose(y2_test, (0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield [x, x2], y


def siamese_generator_test_batch(val_txt, batch_size, num_classes, img_path):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        # random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            # y_test, y_labels = process_batch(new_line[a:b], img_path, train=False)
            y_test, y2_test, y_labels = siamese_process_batch(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            x = np.transpose(x, (0, 2, 3, 1, 4))
            x2= np.transpose(y2_test, (0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield [x, x2], y


