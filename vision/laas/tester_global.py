# -*- coding:utf-8 -*-
from models.c3d import c3d_model
import models.models as m
from keras.optimizers import SGD,Adam
from tools.training.schedules import onetenth_4_8_12
import numpy as np
import random
import cv2
import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import argparse
import json
import eval_toolkit

def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser(description='Training models')

    ap.add_argument('-tr', action='store',
                    dest='train_file',
                    help='Training file')
    ap.add_argument('-ts', action='store',
                    dest='test_file',
                    help='Testing file')
    ap.add_argument('-bs', action='store',
                    default=16, type=int,
                    dest='batch_size',
                    help='Batch size')
    ap.add_argument('-eps', action='store',
                    dest='epochs', default=10, type=int,
                    help='Num epochs')
    ap.add_argument('-net_video', action='store', required=True,
                    dest='net_video', type=str,
                    help='CNN to use: [C3D, resnet2D_concat, resnet3D_18, resnet3D_34]')
    ap.add_argument('-df', action='store', required=True,
                    dest='doFlip', type=int,
                    help='Do perform Flip in training?')
    ap.add_argument('-ds', action='store', required=True,
                    dest='doScale', type=int,
                    help='Do perform Scale in training?')
    ap.add_argument('-mp', action='store',
                    dest='model_path', type=str,
                    help='Model base path')
    ap.add_argument('-sp', action='store',
                    dest='save_path', type=str,
                    help='Save path')

    ap.add_argument('--gpu', default='', type=str)
    ap.add_argument('--resume', default=r'pretrained/weights.h5', type=str)
    ap.add_argument('--data_path', default='4persons', type=str)
    # set up network configuration.
    ap.add_argument('--net_audio', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
    ap.add_argument('--ghost_cluster', default=2, type=int)
    ap.add_argument('--vlad_cluster', default=8, type=int)
    ap.add_argument('--bottleneck_dim', default=512, type=int)
    ap.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
    # set up learning rate, training loss and optimizer.
    ap.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
    ap.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
    ap.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], type=str)

    args = ap.parse_args()

    args.doFlip  = bool(args.doFlip)
    args.doScale = bool(args.doScale)

    if args.train_file is None:
        # ### ibug-avs heads
        # args.train_file = 'lists/ibug-avs/train_heads_list_angle_c10.txt'
        # args.test_file  = 'lists/ibug-avs/test_heads_list_angle_c10.txt'
        #
        # args.train_file = 'lists/ibug-avs/train_heads_list_angle_c5.txt'
        # args.test_file  = 'lists/ibug-avs/test_heads_list_angle_c5.txt'
        #
        # args.train_file = 'lists/ibug-avs/train_heads_list_angle.txt'
        # args.test_file  = 'lists/ibug-avs/test_heads_list_angle.txt'
        #
        # args.train_file = 'lists/ibug-avs/train_heads_list_angle_c6.txt'
        # args.test_file  = 'lists/ibug-avs/test_heads_list_angle_c6.txt'

        #### lr-vid-ict lips
        # args.train_file = 'lists/train_heads_list_lr-vid-ict-ibugs.txt'
        # args.test_file  = 'lists/test_heads_list_lr-vid-ict-ibugs.txt'

        #### amicorpus
        args.train_file = 'lists/amicorpus/train_heads_list_binary.txt'
        args.test_file  = 'lists/amicorpus/test_heads_list_binary.txt'
        args.model_path = 'weights/ibugs/siamese/heads_angles_v2_optflow/'


    if args.save_path is None:
        # ### ibug-avs heads
        args.save_path = 'results/ibugs/heads_angles/'
        #### lr-vid-ict lips
        args.save_path = 'results/ibugs/heads_list_lr-vid-ict-ibugs/'
        # args.save_path = 'weights/lr-vid-ict/lips/'
        #### amicorpus
        args.save_path = 'results/amicorpus/heads/'

    num_classes = 0
    with open(args.test_file, 'r') as infile:
        for line in infile:
            num = int(line.split(' ')[-1]) + 1
            if num > num_classes:num_classes = num

    #####Tmp
    num_classes = 10

    save_path = args.save_path

    conf = ''
    if args.doFlip: conf += '_Flipped'
    if args.doScale:conf += '_Scaled'

    model_path = args.model_path + '/' + args.net_video + '_' + args.net_audio + '/' + 'C' + str(num_classes) + 'E' + str(args.epochs) + conf + '/'
    save_path = args.save_path   + '/' + args.net_video + '_' + args.net_audio + '/' + 'C' + str(num_classes) + 'E' + str(
        args.epochs) + conf + '/'

    trained_model, generator_train_batch, generator_val_batch, generator_test_batch = \
        load_model(args.net_video,
                   num_classes, model_path, args.test_file, args.train_file,
                   args.doFlip, args.doScale)


    eval_toolkit.evaluate_model(args.test_file, trained_model, generator_test_batch, save_path, num_classes, 16)


    with open(save_path + 'commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    print(args)
    print('Save path :' + args.save_path)


def load_model( network_video, num_classes, model_path, val_file, train_file, do_flip=True, do_scale=True):

    m.doFlip = do_flip
    m.doScale = do_scale

    # model = c3d_model(num_classes, kernel_w, kernel_h)
    if 'resnet3D_18' == network_video:
        model, generator_train_batch, generator_val_batch, generator_test_batch = m.r3d_18(num_classes)
    elif 'resnet3D_34' == network_video:
        model, generator_train_batch, generator_val_batch, generator_test_batch = m.r3d_34(num_classes)
    elif 'resnet2D_concat' == network_video:
        model, generator_train_batch, generator_val_batch, generator_test_batch = m.r2d_50(num_classes)
    elif 'C3D' == network_video:
        model, generator_train_batch, generator_val_batch, generator_test_batch = m.c3d(num_classes)
    elif 'sC3D' == network_video:
        model, generator_train_batch, generator_val_batch, generator_test_batch = m.siamese_c3d(num_classes)
    elif 'sresnet3D_18' == network_video:
        model, generator_train_batch, generator_val_batch, generator_test_batch = m.siamese_r3d_18(num_classes)
    elif 'sresnet3D_34' == network_video:
        model, generator_train_batch, generator_val_batch, generator_test_batch = m.siamese_r3d_34(num_classes)
    else:
        raise Exception('No valid network.')

    model.summary()
    model.load_weights(model_path + '/weights.h5')
    return model, generator_train_batch, generator_val_batch, generator_test_batch

def get_num_samples(test_file, train_file):
    f1 = open(train_file, 'r')
    f2 = open(test_file, 'r')
    lines = f1.readlines()
    f1.close()
    train_samples = len(lines)
    lines = f2.readlines()
    f2.close()
    val_samples = len(lines)
    return train_samples, val_samples


if __name__ == '__main__':
    main()
    exit()