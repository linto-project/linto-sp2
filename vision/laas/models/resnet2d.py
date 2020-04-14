from keras import applications
from keras.layers import Dense, Dropout,GlobalAveragePooling2D
from keras.models import Model
from keras.utils import np_utils
import numpy as np
import glob
import random
import cv2

kernel_w = 224
kernel_h = 224
img_w = 320 # 171
img_h = 240 # 128
clip_size = 16
h_start = (img_h-kernel_h)//2
h_end   = h_start + kernel_h
w_start = (img_w-kernel_w)//2
w_end   = w_start + kernel_w

doFlip = True
doScale= True

def resnet50_model(num_classes,img_w, img_h, channels = 3):
    base_model = applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=(img_h, img_w, channels))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def process_batch(lines,img_path,train=True):
    num = len(lines)
    batch = np.zeros((num, kernel_w, kernel_h, 3*clip_size), dtype='float32')
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
        if (symbol+clip_size) >= len(imgs):
            continue
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)

            sc = random.uniform(0.2, 1)

            tmp_concat_img = []
            for j in range(clip_size):
                crop_x = random.randint(0, 15) # add detection noise
                crop_y = random.randint(0, 58)
                img = imgs[symbol + j]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
                if doScale:
                    # Re-scale
                    hh, ww = image.shape[0:2]
                    hh = int(hh * sc)
                    ww = int(ww * sc)
                    image = cv2.resize(image, (ww, hh))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (img_w, img_h))
                if is_flip == 1 and doFlip:
                    image = cv2.flip(image, 1)
                if tmp_concat_img == []:
                    tmp_concat_img = image[crop_x:crop_x + kernel_w, crop_y:crop_y + kernel_h, :]
                else:
                    tmp_concat_img = np.concatenate((tmp_concat_img, image[crop_x:crop_x + kernel_w, crop_y:crop_y + kernel_h, :]), axis=2)
            batch[i][:][:][:] = tmp_concat_img
            labels[i] = label
        else:
            tmp_concat_img = []
            for j in range(cli.p_size):
                img = imgs[symbol + j]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (img_w, img_h))
                if tmp_concat_img == []:
                    tmp_concat_img = image[h_start:h_end, w_start:w_end, :]
                else:
                    tmp_concat_img = np.concatenate((tmp_concat_img, image[h_start:h_end, w_start:w_end, :]), axis=2)
            batch[i][:][:][:] = tmp_concat_img
            labels[i] = label
    return batch, labels


def process_batch_org(lines,img_path,train=True):
    num = len(lines)
    batch = np.zeros((num, kernel_w, kernel_h, 3*clip_size), dtype='float32')
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
        if (symbol+clip_size) >= len(imgs):
            continue
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)

            sc = random.uniform(0.2, 1)

            tmp_concat_img = []
            for j in range(clip_size):
                img = imgs[symbol + j]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
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
                if tmp_concat_img == []:
                    tmp_concat_img = image[crop_x:crop_x + kernel_w, crop_y:crop_y + kernel_h, :]
                else:
                    tmp_concat_img = np.concatenate((tmp_concat_img, image[crop_x:crop_x + kernel_w, crop_y:crop_y + kernel_h, :]), axis=2)
            batch[i][:][:][:] = tmp_concat_img
            labels[i] = label
        else:
            tmp_concat_img = []
            for j in range(clip_size):
                img = imgs[symbol + j]
                # image = cv2.imread(img_path + path + '/' + img)
                image = cv2.imread(img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (img_w, img_h))
                if tmp_concat_img == []:
                    tmp_concat_img = image[h_start:h_end, w_start:w_end, :]
                else:
                    tmp_concat_img = np.concatenate((tmp_concat_img, image[h_start:h_end, w_start:w_end, :]), axis=2)
            batch[i][:][:][:] = tmp_concat_img
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


def generator_train_batch(train_txt,batch_size,num_classes,img_path):
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
            x_train, x_labels = process_batch(new_line[a:b],img_path,train=True)
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            # x = np.transpose(x, (0,2,3,1,4))
            x = np.transpose(x, (0,1,2,3))
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
            y_test,y_labels = process_batch(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            # x = np.transpose(x, (0,2,3,1,4))
            x = np.transpose(x, (0,1,2,3))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def generator_test_batch(val_txt,batch_size,num_classes,img_path):
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
            y_test,y_labels = process_batch(new_line[a:b],img_path,train=False)
            x = preprocess(y_test)
            # x = np.transpose(x, (0,2,3,1,4))
            x = np.transpose(x, (0,1,2,3))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y

