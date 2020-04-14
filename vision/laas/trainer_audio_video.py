# -*- coding:utf-8 -*-import os
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import models.models as m
from keras.optimizers import SGD,Adam
import keras.backend as K
import pickle
from tools.training.schedules import onetenth_4_8_12, wideresnet_step
import numpy as np
import random
import cv2
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import argparse
import json
import eval_toolkit
from tools.utils import get_num_samples


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
                    help='CNN to use: [C3D, resnet2D_concat, resnet3D_18, resnet3D_34, sC3D, sresnet3D_18]')
    ap.add_argument('-df', action='store', required=True,
                    dest='doFlip', type=int,
                    help='Do perform Flip in training?')
    ap.add_argument('-ds', action='store', required=True,
                    dest='doScale', type=int,
                    help='Do perform Scale in training?')
    ap.add_argument('-sp', action='store',
                    dest='save_path', type=str,
                    help='Save path')
    ap.add_argument('-lt', action='store', default=0,
                    dest='load_opt_train', type=int,
                    help='Load the optimizer')
    ap.add_argument('-st', action='store', default=0,
                    dest='save_opt_train', type=int,
                    help='Save the optimizer')
    ap.add_argument('-eval', default='yes', choices=['yes', 'no'], type=str)


    ap.add_argument('-gpu', default='', type=str)
    ap.add_argument('-resume', default=r'pretrained/weights.h5', type=str)
    ap.add_argument('-data_path', default='4persons', type=str)
    # set up network configuration.
    ap.add_argument('-net_audio', default='resnet34s', choices=['resnet34s', 'resnet34l', 'None'], type=str)
    ap.add_argument('-ghost_cluster', default=2, type=int)
    ap.add_argument('-vlad_cluster', default=8, type=int)
    ap.add_argument('-bottleneck_dim', default=512, type=int)
    ap.add_argument('-aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
    # set up learning rate, training loss and optimizer.
    ap.add_argument('-loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
    ap.add_argument('-test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)
    ap.add_argument('-optimizer', default='adam', choices=['adam', 'sgd'], type=str)

    args = ap.parse_args()

    args.doFlip  = bool(args.doFlip)
    args.doScale = bool(args.doScale)

    # ==================================
    #       Get Model
    # ==================================
    # construct the data generator.
    # params = {'dim': (257, None, 1),
    #           'nfft': 512,
    #           'min_slice': 720,
    #           'win_length': 400,
    #           'hop_length': 160,
    #           'n_classes': 5994,
    #           'sampling_rate': 16000,
    #           'normalize': True,
    #           }
    # model = m.siamese_c3d(2)
    # model = m.video_audio(input_dim=params['dim'],
    #                     num_class=params['n_classes'],
    #                     mode='train', args=args)
    # return

    if args.train_file is None or args.train_file == 'None':
        #### amicorpus
        # args.train_file = 'lists/amicorpus/train_heads_list_cv1IS1002b.txt'
        # args.test_file  = 'lists/amicorpus/test_heads_list_cv1IS1002b.txt'
        # args.train_file = 'lists/amicorpus/train_heads_list_cv125_close12.txt'
        # args.test_file  = 'lists/amicorpus/test_heads_list_cv125_close12.txt'
        args.train_file = "lists/amicorpus/train_heads_list_cv5.txt"
        args.test_file = "lists/amicorpus/test_heads_list_cv5.txt"

    if args.save_path is None or args.save_path == 'None':
        ### amicorpus
        # args.save_path = "weights/ami/"
        args.save_path = "weights/ami_ws_cv5/"

    num_classes = 0
    with open(args.test_file, 'r') as infile:
        for line in infile:
            num = int(line.split(' ')[-1]) + 1
            if num > num_classes:num_classes = num

    conf = ''
    if args.doFlip: conf += '_Flipped'
    if args.doScale:conf += '_Scaled'

    args.save_path = args.save_path + '/' + args.net_video
    if not args.net_audio == 'None':
        args.save_path = args.save_path + '_' + args.net_audio
    args.save_path = args.save_path + '/' + 'C' + str(num_classes) + 'E' + str(args.epochs) + conf + '/'

    args.num_classes = num_classes
    trained_model, generator_train_batch, generator_val_batch, generator_test_batch = \
        train_model(args)

    # train_model(args.batch_size, args.epochs, args.net_video, args.net_audio,
    #             args.num_classes, args.save_path, args.test_file, args.train_file,
    #             args.doFlip, args.doScale)

    print('Save path :' + args.save_path)
    print(args)

    with open(args.save_path + 'commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.eval == 'yes':
        eval_toolkit.evaluate_model(args.test_file, trained_model, generator_test_batch, args.save_path + '/test/', args.num_classes, 16)

# def train_model(batch_size, epochs, network_video, network_audio, num_classes, save_path, val_file, train_file, do_flip=True, do_scale=True):

def train_model(args):
    train_samples, val_samples = get_num_samples(args.test_file, args.train_file)
    img_path = ''

    m.doFlip = args.doFlip
    m.doScale = args.doScale

    generator_test_batch, generator_train_batch, generator_val_batch, model = m.get_model(args)

    if os.path.isfile(args.save_path + '/weights.h5') and args.load_opt_train == 0:
        # model.summary()
        model.load_weights(args.save_path+ '/weights.h5')
        return model, generator_train_batch, generator_val_batch, generator_test_batch

    lr = 0.0005 # org
    lr = 0.05 # OK for C3D
    # lr = 0.005
    # opt = SGD(lr=lr, momentum=0.9, nesterov=True)
    opt = Adam(lr=lr, decay=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # model.summary()

    # Save optimazer state
    if args.load_opt_train == 1:
        model.load_weights(args.save_path + '/weights.h5')
        model._make_train_function()
        with open(args.save_path + 'optimizer.pkl', 'rb') as f:
            weight_values = pickle.load(f)
        model.optimizer.set_weights(weight_values)

    print(args)
    history = model.fit_generator(generator_train_batch(args.train_file, args.batch_size, args.num_classes, img_path),
                                  steps_per_epoch=train_samples // args.batch_size,
                                  epochs=args.epochs,
                                  # callbacks=[onetenth_4_8_12(lr)],
                                  validation_data=generator_val_batch(args.test_file,
                                                                      args.batch_size, args.num_classes, img_path),
                                  validation_steps=val_samples // args.batch_size,
                                  verbose=1)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    plot_history(history, args.save_path)
    save_history(history, args.save_path)
    # Save model
    model.save_weights(args.save_path + '/weights.h5')

    # Save optimazer state
    if args.save_opt_train == 1:
        symbolic_weights = getattr(model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open(args.save_path + '/optimizer.pkl', 'wb') as f:
            pickle.dump(weight_values, f)

    with open(args.save_path + '/settings.txt', 'w') as outfile:
        outfile.write('num_classes:\t' + str(args.num_classes) + '\n')
        outfile.write('batch_size:\t' + str(args.batch_size) + '\n')
        outfile.write('epochs:\t' + str(args.epochs) + '\n')
        outfile.write('img_path:\t' + str(img_path) + '\n')
        outfile.write('train_file:\t' + str(args.train_file) + '\n')
        outfile.write('test_file:\t' + str(args.test_file) + '\n')
        outfile.write('lr:\t' + str(lr) + '\n')

    return model, generator_train_batch, generator_val_batch, generator_test_batch





if __name__ == '__main__':
    main()
    exit()