from keras.layers import Dense,Dropout,Conv3D,Input,MaxPool3D,Flatten,Activation,concatenate
from keras.regularizers import l2
from keras.models import Model

from keras.utils import np_utils
import numpy as np
import glob
import random
import cv2
from tools.utils import noisy

kernel_w = 112
kernel_h = 112
img_w = 171
img_h = 128

doFlip = True
doScale= True
doTrans= True
doNoisy= True


def c3d_model(nb_classes = 101, img_w=112, img_h=112, num_channels=3, include_top=True):
    input_shape = (img_w, img_h,16,num_channels)
    weight_decay = 0.005

    inputs = Input(input_shape)
    x = Conv3D(64,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(inputs)
    x = MaxPool3D((2,2,1),strides=(2,2,1),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(128,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256,(3,3,3),strides=(1,1,1),padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2,2,2),strides=(2,2,2),padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu',kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Flatten()(x)

    if include_top:
        x = Dense(2048, activation='relu', kernel_regularizer=l2(weight_decay))(x)
        x = Dropout(0.5)(x)
        x = Dense(2048, activation='relu', kernel_regularizer=l2(weight_decay))(x)
        x = Dropout(0.5)(x)
        x = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(x)
        x = Activation('softmax')(x)

    model = Model(inputs, x)
    return model


def siamese_c3d(nb_classes = 101, img_w=112, img_h=112, include_top=True):

    # x = c3d_model(nb_classes=nb_classes, img_w=img_w, img_h=img_h, num_channels=3, include_top=True)
    # return x

    weight_decay = 0.005
    x = c3d_model(nb_classes=nb_classes, img_w=img_w, img_h=img_h, num_channels=3, include_top=False)
    y = c3d_model(nb_classes=nb_classes, img_w=img_w, img_h=img_h, num_channels=3, include_top=False)

    z = concatenate([x.output, y.output])

    if include_top:
        z = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(z)
        z = Dropout(0.5)(z)
        z = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(z)
        z = Dropout(0.5)(z)
        z = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(z)
        z = Activation('softmax')(z)

    model = Model(inputs=[x.input, y.input], outputs=z)
    return model


    input_shape = (img_w, img_h, 16, 3)
    weight_decay = 0.0005
    inputx = Input(input_shape)
    x = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(inputx)
    x = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    # x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    x = Flatten()(x)

    # x = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(x)
    # x = Dropout(0.5)(x)

    inputy = Input(input_shape)
    y = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(inputy)
    y = MaxPool3D((2, 2, 1), strides=(2, 2, 1), padding='same')(y)

    y = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(y)
    y = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(y)

    y = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(y)
    # y = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(y)

    y = Conv3D(256, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(y)
    y = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(y)

    y = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(y)

    y = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(y)
    y = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(y)

    y = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(y)

    y = Conv3D(512, (3, 3, 3), strides=(1, 1, 1), padding='same',
               activation='relu', kernel_regularizer=l2(weight_decay))(y)
    y = MaxPool3D((2, 2, 2), strides=(2, 2, 2), padding='same')(y)
    y = Flatten()(y)

    # y = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(y)
    # y = Dropout(0.5)(y)

    from keras.layers import add, average
    # combine the output of the two branches
    z = concatenate([x, y])

    # z = Flatten()(z)
    z = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(z)
    z = Dropout(0.5)(z)
    z = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(z)
    z = Dropout(0.5)(z)
    z = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(z)
    z = Activation('softmax')(z)

    model = Model(inputs=[inputx, inputy], outputs=z)
    return model


def siamese_c3dorg(nb_classes = 101, img_w=112, img_h=112):
    input_shape = (img_w, img_h, 16, 3)
    weight_decay = 0.005

    x = c3d_model(nb_classes=nb_classes, img_w=img_w, img_h=img_h, num_channels=3, include_top=False)
    y = c3d_model(nb_classes=nb_classes, img_w=img_w, img_h=img_h, num_channels=3, include_top=False)

    # combine the output of the two branches
    z = concatenate([x.output, y.output])

    # z = Flatten()(z)
    z = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(z)
    z = Dropout(0.1)(z)
    z = Dense(2048,activation='relu',kernel_regularizer=l2(weight_decay))(z)
    # z = Dropout(0.5)(z)
    z = Dense(nb_classes,kernel_regularizer=l2(weight_decay))(z)
    z = Activation('softmax')(z)

    model = Model(inputs=[x.input, y.input], outputs=z)
    return model


def process_batch_rgb(lines,img_path,train=True):
    num = len(lines)
    batch = np.zeros((num,16,112,112,3),dtype='float32')
    labels = np.zeros(num,dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol)-1
        # imgs = os.listdir(img_path+path)
        imgs = glob.glob(img_path+path+'/*.jpg')
        if imgs == []:
            imgs = glob.glob(img_path+path+'/*.png')
        imgs = [k for k in imgs if '.wav' not in k and 'flow' not in k]
        imgs.sort(key=str.lower)
        if (symbol+16) >= len(imgs):
            continue
        if train:
            crop_x = random.randint(0, 10)
            crop_y = random.randint(0, 10)
            tran_x = random.randint(0, 10)
            tran_y = random.randint(0, 10)
            is_flip = random.randint(0, 1)
            init_f = random.randint(0, 8)
            init_f = 0
            if (symbol+16+init_f) >= len(imgs):
                init_f = 0

            sc = random.uniform(0.2, 1)

            for j in range(16):
                img = imgs[symbol + j + init_f]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
                # Re-scale
                if doScale:
                    hh, ww = image.shape[0:2]
                    hh = int(hh * sc)
                    ww = int(ww * sc)
                    image = cv2.resize(image, (ww, hh))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                if is_flip == 1 and doFlip:
                    image = cv2.flip(image, 1)

                if doTrans:
                    hh, ww = image.shape[0:2]
                    M = np.float32([[1, 0, tran_x], [0, 1, tran_y]])
                    image = cv2.warpAffine(image, M, (ww, hh))

                if doNoisy:
                    noisy(image, 'gauss')

                batch[i][j][:][:][:] = image[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
            labels[i] = label
        else:
            for j in range(16):
                img = imgs[symbol + j]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                batch[i][j][:][:][:] = image[8:120, 30:142, :]
            labels[i] = label
    return batch, labels


def process_batch_opt_flow(lines,img_path,train=True):
    num = len(lines)
    btcof = np.zeros((num,16,112,112,3),dtype='float32')
    labels = np.zeros(num,dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol)-1
        # imgs = os.listdir(img_path+path)
        imgs = glob.glob(img_path+path+'/*.jpg')
        if imgs == []:
            imgs = glob.glob(img_path+path+'/*.png')
        imgs.sort(key=str.lower)
        if (symbol+16) >= len(imgs):
            continue
        if train:
            crop_x = random.randint(0, 10)
            crop_y = random.randint(0, 10)
            tran_x = random.randint(0, 10)
            tran_y = random.randint(0, 10)
            is_flip = random.randint(0, 1)
            init_f = random.randint(0, 8)
            init_f = 0
            if (symbol+16+init_f) >= len(imgs):
                init_f = 0

            sc = random.uniform(0.2, 1)

            for j in range(16):
                img = imgs[symbol + j + init_f]
                im_of = np.load(img + '.npy')
                im_of *= 16
                im_of = cv2.normalize(im_of, None, 0, 255, cv2.NORM_MINMAX)

                # Re-scale
                if doScale:
                    hh, ww = im_of.shape[0:2]
                    hh = int(hh * sc)
                    ww = int(ww * sc)
                    im_of = cv2.resize(im_of, (ww, hh))

                # image = cv2.resize(image, (171, 128))
                if is_flip == 1 and doFlip:
                    im_of = cv2.flip(im_of, 1)

                if doTrans:
                    hh, ww = im_of.shape[0:2]
                    M = np.float32([[1, 0, tran_x], [0, 1, tran_y]])
                    im_of = cv2.warpAffine(im_of, M, (ww, hh))

                if doNoisy:
                    noisy(im_of, 'gauss')

                btcof[i][j][:][:][:] = im_of[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
            labels[i] = label
        else:
            for j in range(16):
                img = imgs[symbol + j]
                # image = cv2.imread(img_path + path + '/' + img)
                im_of = np.load(img + '.npy')
                btcof[i][j][:][:][:] = im_of[8:120, 30:142, :]
            labels[i] = label
    return btcof, labels

def process_batch_org(lines,img_path,train=True):
    num = len(lines)
    batch = np.zeros((num,16,112,112,3),dtype='float32')
    labels = np.zeros(num,dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol)-1
        # imgs = os.listdir(img_path+path)
        imgs = glob.glob(img_path+path+'/*.jpg')
        if imgs == []:
            imgs = glob.glob(img_path+path+'/*.png')
        imgs.sort(key=str.lower)
        if (symbol+16) >= len(imgs):
            continue
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)
            init_f = random.randint(0, 8)
            init_f = 0
            if (symbol+16+init_f) >= len(imgs):
                init_f = 0
            # if label == 1:
            sc = random.uniform(0.2, 1)

            for j in range(16):
                img = imgs[symbol + j + init_f]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
                # Re-scale
                if doScale:
                    hh, ww = image.shape[0:2]
                    hh = int(hh * sc)
                    ww = int(ww * sc)
                    image = cv2.resize(image, (ww, hh))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                if is_flip == 1 and doFlip:
                    image = cv2.flip(image, 1)
                batch[i][j][:][:][:] = image[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
            labels[i] = label
        else:
            for j in range(16):
                img = imgs[symbol + j]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                batch[i][j][:][:][:] = image[8:120, 30:142, :]
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
        for i in range(int(num/batch_size)):
            a = i*batch_size
            b = (i+1)*batch_size
            x_train, x_labels = process_batch_rgb(new_line[a:b],img_path,train=True)
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            x = np.transpose(x, (0,2,3,1,4))
            yield x, y


def generator_val_batch(val_txt,batch_size,num_classes,img_path):
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
            y_test,y_labels = process_batch_rgb(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def generator_test_batch(val_txt,batch_size,num_classes,img_path):
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
            y_test,y_labels = process_batch_rgb(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def generator_train_batch_opt_flow(train_txt,batch_size,num_classes,img_path):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num/batch_size)):
            a = i*batch_size
            b = (i+1)*batch_size
            x_train, x_labels = process_batch_opt_flow(new_line[a:b],img_path,train=True)
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            x = np.transpose(x, (0,2,3,1,4))
            yield x, y


def generator_val_batch_opt_flow(val_txt,batch_size,num_classes,img_path):
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
            y_test,y_labels = process_batch_opt_flow(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def generator_test_batch_opt_flow(val_txt,batch_size,num_classes,img_path):
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
            y_test,y_labels = process_batch_opt_flow(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def siamese_process_batch_rgb(lines,img_path,train=True):
    num = len(lines)
    batch = np.zeros((num,16,112,112,3),dtype='float32')
    btcof = np.zeros((num,16,112,112,3),dtype='float32')
    labels = np.zeros(num,dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol)-1
        # imgs = os.listdir(img_path+path)
        imgs = glob.glob(img_path+path+'/*.jpg')
        if imgs == []:
            imgs = glob.glob(img_path+path+'/*.png')
        imgs.sort(key=str.lower)
        if (symbol+16) >= len(imgs):
            continue
        if train:
            crop_x = random.randint(0, 10)
            crop_y = random.randint(0, 10)
            tran_x = random.randint(0, 10)
            tran_y = random.randint(0, 10)
            is_flip = random.randint(0, 1)
            init_f = random.randint(0, 8)
            init_f = 0
            if (symbol+16+init_f) >= len(imgs):
                init_f = 0

            sc = random.uniform(0.2, 1)

            for j in range(16):
                img = imgs[symbol + j + init_f]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                im_of = np.load(img + '.npy')
                mag, ang = cv2.cartToPolar(im_of[..., 0], im_of[..., 1])
                im_of = np.dstack((im_of, mag))
                # im_of *= 16
                im_of = cv2.normalize(im_of, None, -1, 1, cv2.NORM_MINMAX)

                # Re-scale
                if doScale:
                    hh, ww = image.shape[0:2]
                    hh = int(hh * sc)
                    ww = int(ww * sc)
                    image = cv2.resize(image, (ww, hh))
                    im_of = cv2.resize(im_of, (ww, hh))

                # image = cv2.resize(image, (171, 128))
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

                # cv2.imshow('opticalflow',im_of[crop_x:crop_x + 112, crop_y:crop_y + 112, :])
                # cv2.waitKey(1)

                batch[i][j][:][:][:] = image[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
                btcof[i][j][:][:][:] = im_of[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
            labels[i] = label
        else:
            for j in range(16):
                img = imgs[symbol + j]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                im_of = np.load(img + '.npy')
                mag, ang = cv2.cartToPolar(im_of[..., 0], im_of[..., 1])
                im_of = np.dstack((im_of, mag))
                im_of = cv2.normalize(im_of, None, -1, 1, cv2.NORM_MINMAX)
                batch[i][j][:][:][:] = image[8:120, 30:142, :]
                btcof[i][j][:][:][:] = im_of[8:120, 30:142, :]
            labels[i] = label
    return batch, btcof, labels


def siamese_process_batch_v1(lines,img_path,train=True):
    binarize = False
    num = len(lines)
    batch = np.zeros((num,16,112,112,3),dtype='float32')
    btcof = np.zeros((num,16,112,112,3),dtype='float32')
    labels = np.zeros(num,dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol)-1
        # if binarize:
        #     label = 0 if label < 5 else 1
        # imgs = os.listdir(img_path+path)
        imgs = glob.glob(img_path+path+'/*.jpg')
        if imgs == []:
            imgs = glob.glob(img_path+path+'/*.png')
        imgs.sort(key=str.lower)
        if (symbol+16) >= len(imgs):
            continue
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)
            init_f = random.randint(0, 8)
            init_f = 0
            if (symbol+16+init_f) >= len(imgs):
                init_f = 0

            sc = random.uniform(0.2, 1)

            for j in range(16):
                img_path = imgs[symbol + j + init_f]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img_path)
                im_of = np.load(img_path + '.npy')
                # mag, ang = cv2.cartToPolar(im_of[..., 0], im_of[..., 1])
                # im_of = np.dstack((im_of, mag))
                im_of *= 16
                im_of = cv2.normalize(im_of, None, 0, 255, cv2.NORM_MINMAX)
                # Re-scale
                if doScale:
                    hh, ww = image.shape[0:2]
                    hh = int(hh * sc)
                    ww = int(ww * sc)
                    image = cv2.resize(image, (ww, hh))
                    im_of = cv2.resize(im_of, (ww, hh))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                im_of = cv2.resize(im_of, (171, 128))

                if is_flip == 1 and doFlip:
                    image = cv2.flip(image, 1)
                    im_of = cv2.flip(im_of, 1)
                batch[i][j][:][:][:] = image[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
                btcof[i][j][:][:][:] = im_of[crop_x:crop_x + 112, crop_y:crop_y + 112, :]
            labels[i] = label
        else:
            for j in range(16):
                img_path = imgs[symbol + j]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img_path)
                im_of = np.load(img_path + '.npy')
                # mag, ang = cv2.cartToPolar(im_of[..., 0], im_of[..., 1])
                # im_of = np.dstack((im_of, mag))
                im_of *= 16
                im_of = cv2.normalize(im_of, None, 0, 255, cv2.NORM_MINMAX)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                im_of = cv2.resize(im_of, (171, 128))

                batch[i][j][:][:][:] = image[8:120, 30:142, :]
                btcof[i][j][:][:][:] = im_of[8:120, 30:142, :]
            labels[i] = label
    tmp = np.zeros((num, 16, 112, 112, 3), dtype='float32')
    return batch, batch, labels


def siamese_generator_train_batch(train_txt,batch_size,num_classes,img_path):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num/batch_size)):
            a = i*batch_size
            b = (i+1)*batch_size
            x_train, x2_train, x_labels = siamese_process_batch_rgb(new_line[a:b],img_path,train=True)
            # x_train, x_labels = process_batch_rgb(new_line[a:b],img_path,train=True)
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            x = np.transpose(x, (0,2,3,1,4))
            x2= np.transpose(x2_train, (0,2,3,1,4))
            # yield x, y
            yield [x, x2], y


def siamese_generator_val_batch(val_txt,batch_size,num_classes,img_path):
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
            y_test, y2_test, y_labels = siamese_process_batch_rgb(new_line[a:b],img_path,train=False)
            # y_test, y_labels = process_batch_rgb(new_line[a:b],img_path,train=True)
            x = preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            x2= np.transpose(y2_test, (0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield [x, x2], y


def siamese_generator_test_batch(val_txt,batch_size,num_classes,img_path):
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
            y_test, y2_test, y_labels = siamese_process_batch_rgb(new_line[a:b],img_path,train=False)
            # y_test, y_labels = process_batch_rgb(new_line[a:b],img_path,train=True)
            x = preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            x2= np.transpose(y2_test, (0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield [x, x2], y




