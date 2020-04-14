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
import cv2
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


def siamese_r3d_18(nb_classes, img_w, img_h, include_top=True):

    input_shape = (img_w, img_h, 16, 3)
    x = Resnet3DBuilder.build_resnet_18(input_shape, -1)
    y = Resnet3DBuilder.build_resnet_18(input_shape, -1)

    z = concatenate([x.output, y.output])

    if include_top:
        z = Dense(units=nb_classes,
                  kernel_initializer="he_normal",
                  activation="softmax",
                  kernel_regularizer=l2(1e-4))(z)

    model = Model(inputs=[x.input, y.input], outputs=z)
    return model


def siamese_r3d_34(nb_classes, img_w, img_h, include_top=True):

    input_shape = (img_w, img_h, 16, 3)
    x = Resnet3DBuilder.build_resnet_34(input_shape, -1)
    y = Resnet3DBuilder.build_resnet_34(input_shape, -1)

    z = concatenate([x.output, y.output])

    if include_top:
        z = Dense(units=nb_classes,
                  kernel_initializer="he_normal",
                  activation="softmax",
                  kernel_regularizer=l2(1e-4))(z)

    model = Model(inputs=[x.input, y.input], outputs=z)
    return model


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
        imgs = glob.glob(img_path + path + '/*.jpg')
        if imgs == []:
            imgs = glob.glob(img_path + path + '/*.png')
        imgs.sort(key=str.lower)
        if (symbol + 16) >= len(imgs):
            continue
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
                # crop_x = random.randint(0, 15)  # add detection noise
                # crop_y = random.randint(0, 58)
                img = imgs[symbol + j]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # image = cv2.resize(image, (img_w, img_h))

                im_of = np.load(img + '.npy') if isfile(img + '.npy') else np.load(img[:-4] + '.npz')['flow']
                if im_of.shape[2] == 2:
                    mag, ang = cv2.cartToPolar(im_of[..., 0], im_of[..., 1])
                    im_of = np.dstack((im_of, mag))
                # im_of *= 16
                im_of = cv2.normalize(im_of, None, -1, 1, cv2.NORM_MINMAX)
                # im_of = cv2.resize(im_of, (img_w, img_h))

                image = cv2.resize(image, (kernel_w, kernel_h))
                im_of = cv2.resize(im_of, (kernel_w, kernel_h))
                batch[i][j][:][:][:] = image
                btcof[i][j][:][:][:] = im_of
                continue

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

                batch[i][j][:][:][:] = image[crop_x:crop_x + kernel_w, crop_y:crop_y + kernel_h, :]
                btcof[i][j][:][:][:] = im_of[crop_x:crop_x + kernel_w, crop_y:crop_y + kernel_h, :]
            labels[i] = label
        else:
            for j in range(16):
                img = imgs[symbol + j]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                im_of = np.load(img + '.npy') if isfile(img + '.npy') else np.load(img[:-4] + '.npz')['flow']
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


def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu3D(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault(
        "kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        conv = Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv3d(**conv_params):
    """Helper to build a  BN -> relu -> conv3d block."""
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer",
                                                "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer",
                                                l2(1e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv3D(filters=filters, kernel_size=kernel_size,
                      strides=strides, kernel_initializer=kernel_initializer,
                      padding=padding,
                      kernel_regularizer=kernel_regularizer)(activation)
    return f


def _shortcut3d(input, residual):
    """3D shortcut to match input and residual and merges them with "sum"."""
    stride_dim1 = input._keras_shape[DIM1_AXIS] \
        // residual._keras_shape[DIM1_AXIS]
    stride_dim2 = input._keras_shape[DIM2_AXIS] \
        // residual._keras_shape[DIM2_AXIS]
    stride_dim3 = input._keras_shape[DIM3_AXIS] \
        // residual._keras_shape[DIM3_AXIS]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] \
        == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 \
            or not equal_channels:
        shortcut = Conv3D(
            filters=residual._keras_shape[CHANNEL_AXIS],
            kernel_size=(1, 1, 1),
            strides=(stride_dim1, stride_dim2, stride_dim3),
            kernel_initializer="he_normal", padding="valid",
            kernel_regularizer=l2(1e-4)
            )(input)
    return add([shortcut, residual])


def _residual_block3d(block_function, filters, kernel_regularizer, repetitions,
                      is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            strides = (1, 1, 1)
            if i == 0 and not is_first_layer:
                strides = (2, 2, 2)
            input = block_function(filters=filters, strides=strides,
                                   kernel_regularizer=kernel_regularizer,
                                   is_first_block_of_first_layer=(
                                       is_first_layer and i == 0)
                                   )(input)
        return input

    return f


def basic_block(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
                is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv3D(filters=filters, kernel_size=(3, 3, 3),
                           strides=strides, padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=kernel_regularizer
                           )(input)
        else:
            conv1 = _bn_relu_conv3d(filters=filters,
                                    kernel_size=(3, 3, 3),
                                    strides=strides,
                                    kernel_regularizer=kernel_regularizer
                                    )(input)

        residual = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv1)
        return _shortcut3d(input, residual)

    return f


def bottleneck(filters, strides=(1, 1, 1), kernel_regularizer=l2(1e-4),
               is_first_block_of_first_layer=False):
    """Basic 3 X 3 X 3 convolution blocks. Extended from raghakot's 2D impl."""
    def f(input):
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv3D(filters=filters, kernel_size=(1, 1, 1),
                              strides=strides, padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=kernel_regularizer
                              )(input)
        else:
            conv_1_1 = _bn_relu_conv3d(filters=filters, kernel_size=(1, 1, 1),
                                       strides=strides,
                                       kernel_regularizer=kernel_regularizer
                                       )(input)

        conv_3_3 = _bn_relu_conv3d(filters=filters, kernel_size=(3, 3, 3),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_1_1)
        residual = _bn_relu_conv3d(filters=filters * 4, kernel_size=(1, 1, 1),
                                   kernel_regularizer=kernel_regularizer
                                   )(conv_3_3)

        return _shortcut3d(input, residual)

    return f


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class Resnet3DBuilder(object):
    """ResNet3D."""

    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, reg_factor):
        """Instantiate a vanilla ResNet3D keras model.
        # Arguments
            input_shape: Tuple of input shape in the format
            (conv_dim1, conv_dim2, conv_dim3, channels) if dim_ordering='tf'
            (filter, conv_dim1, conv_dim2, conv_dim3) if dim_ordering='th'
            num_outputs: The number of outputs at the final softmax layer
            block_fn: Unit block to use {'basic_block', 'bottlenack_block'}
            repetitions: Repetitions of unit blocks
        # Returns
            model: a 3D ResNet model that takes a 5D tensor (volumetric images
            in batch) as input and returns a 1D vector (prediction) as output.
        """
        _handle_data_format()
        if len(input_shape) != 4:
            raise ValueError("Input shape should be a tuple "
                             "(conv_dim1, conv_dim2, conv_dim3, channels) "
                             "for tensorflow as backend or "
                             "(channels, conv_dim1, conv_dim2, conv_dim3) "
                             "for theano as backend")

        block_fn = _get_block(block_fn)
        input = Input(shape=input_shape)
        # first conv
        conv1 = _conv_bn_relu3D(filters=64, kernel_size=(7, 7, 7),
                                strides=(2, 2, 2),
                                kernel_regularizer=l2(reg_factor)
                                )(input)
        pool1 = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
                             padding="same")(conv1)

        # repeat blocks
        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block3d(block_fn, filters=filters,
                                      kernel_regularizer=l2(reg_factor),
                                      repetitions=r, is_first_layer=(i == 0)
                                      )(block)
            filters *= 2

        # last activation
        block_output = _bn_relu(block)

        # average poll and classification
        pool2 = AveragePooling3D(pool_size=(block._keras_shape[DIM1_AXIS],
                                            block._keras_shape[DIM2_AXIS],
                                            block._keras_shape[DIM3_AXIS]),
                                 strides=(1, 1, 1))(block_output)
        flatten1 = Flatten()(pool2)
        if num_outputs > 1:
            dense = Dense(units=num_outputs,
                          kernel_initializer="he_normal",
                          activation="softmax",
                          kernel_regularizer=l2(reg_factor))(flatten1)
        elif num_outputs == 0:
            dense = Dense(units=num_outputs,
                          kernel_initializer="he_normal",
                          activation="sigmoid",
                          kernel_regularizer=l2(reg_factor))(flatten1)
        else:
            dense = flatten1

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 18."""
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [2, 2, 2, 2], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 34."""
        return Resnet3DBuilder.build(input_shape, num_outputs, basic_block,
                                     [3, 4, 6, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 50."""
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 4, 6, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 101."""
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 4, 23, 3], reg_factor=reg_factor)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, reg_factor=1e-4):
        """Build resnet 152."""
        return Resnet3DBuilder.build(input_shape, num_outputs, bottleneck,
                                     [3, 8, 36, 3], reg_factor=reg_factor)