# from splitterkit import readwave, writewave, split, merge, combine, slicewave_s
import glob, sys, os
import models.vggvox as vggvox
import models.resnet3d as resnet3d
import argparse
from keras.optimizers import SGD,Adam
from tools.utils import get_num_samples
import json
import eval_toolkit

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
ap.add_argument('-net_video', action='store', default='C3D',
                dest='net_video', type=str,
                help='CNN to use: [C3D, resnet2D_concat, resnet3D_18, resnet3D_34, sC3D, sresnet3D_18]')
ap.add_argument('-df', action='store', default=0,
                dest='doFlip', type=int,
                help='Do perform Flip in training?')
ap.add_argument('-ds', action='store', default=0,
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

#### amicorpus
args.train_file = 'lists/amicorpus/train_heads_list_cv1IS1002b.txt'
args.test_file  = 'lists/amicorpus/test_heads_list_cv1IS1002b.txt'

if args.save_path is None:
    ### amicorpus
    args.save_path = "weights/ami/"

train_samples, val_samples = get_num_samples(args.test_file, args.train_file)
img_path = ''

import models.c3d as m
m.doFlip = args.doFlip
m.doScale = args.doScale


model = vggvox.video_audio(args=args,input_dim=(257, 129, 1), mode='train')

lr = 0.0005 # org
opt = Adam(lr=lr, decay=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

num_classes = 0
with open(args.test_file, 'r') as infile:
    for line in infile:
        num = int(line.split(' ')[-1]) + 1
        if num > num_classes: num_classes = num

conf = ''
if args.doFlip: conf += '_Flipped'
if args.doScale: conf += '_Scaled'

args.save_path = args.save_path + '/' + args.net_video + '_' + args.net_audio + '/' + 'C' + str(
    num_classes) + 'E' + str(args.epochs) + conf + '/'
args.num_classes = num_classes

history = model.fit_generator(vggvox.generator_train_batch_rgb_audio(args.train_file, args.batch_size, args.num_classes, img_path),
                              steps_per_epoch=train_samples // args.batch_size,
                              epochs=args.epochs,
                              # callbacks=[onetenth_4_8_12(lr)],
                              validation_data=vggvox.generator_val_batch_rgb_audio(args.test_file,
                                                                  args.batch_size, args.num_classes, img_path),
                              validation_steps=val_samples // args.batch_size,
                              verbose=1)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
# Save model
model.save_weights(args.save_path + '/weights.h5')

with open(args.save_path + '/settings.txt', 'w') as outfile:
    outfile.write('num_classes:\t' + str(args.num_classes) + '\n')
    outfile.write('batch_size:\t' + str(args.batch_size) + '\n')
    outfile.write('epochs:\t' + str(args.epochs) + '\n')
    outfile.write('img_path:\t' + str(img_path) + '\n')
    outfile.write('train_file:\t' + str(args.train_file) + '\n')
    outfile.write('test_file:\t' + str(args.test_file) + '\n')
    outfile.write('lr:\t' + str(lr) + '\n')

print('Save path :' + args.save_path)
print(args)

with open(args.save_path + 'commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

eval_toolkit.evaluate_model(args.test_file, model, vggvox.generator_test_batch_rgb_audio, args.save_path + '/test/', args.num_classes, 16)





# def main(argv):
#     data_dir = argv[1]
#     txt_dir = argv[2]
#     out_dir = argv[3]
#
#     names = [d for d in os.listdir(txt_dir) if d.endswith('.txt')]
#     for n in names:
#         fichier_audio = n.split("_")[0]
#         spk = 0
#         if os.path.exists(os.path.join(data_dir, fichier_audio, "audio", fichier_audio + ".Mix-Headset.wav")):
#             data = readwave(os.path.join(data_dir, fichier_audio, "audio", fichier_audio + ".Mix-Headset.wav"))
#             txt = open(os.path.join(txt_dir, n), "r")
#             lignes = txt.readlines()
#             for uneLigne in lignes:
#                 champs = uneLigne.split()
#                 test_slice = slicewave_s(data, float(champs[0]), float(champs[1]))
#                 out = os.path.join(out_dir, n.split("_")[0] + "_" + n.split("_")[1] + "/")
#                 writewave(os.path.join(out, str(spk)), test_slice)
#                 spk = spk + 1
#             txt.close()
#         if os.path.exists(os.path.join(data_dir, fichier_audio, "audio", fichier_audio + ".Mix-Lapel.wav")):
#             data = readwave(os.path.join(data_dir, fichier_audio, "audio", fichier_audio + ".Mix-Lapel.wav"))
#             txt2 = open(os.path.join(txt_dir, n), "r")
#             lignes2 = txt2.readlines()
#             for uneLigne2 in lignes2:
#                 champs2 = uneLigne2.split()
#                 test_slice = slicewave_s(data, float(champs2[0]), float(champs2[1]))
#                 out = os.path.join(out_dir, n.split("_")[0] + "_" + n.split("_")[1] + "/")
#                 writewave(os.path.join(out, str(spk)), test_slice)
#                 spk = spk + 1
#             txt2.close()
#
#
# if __name__ == '__main__':
#     main(sys.argv)

# import librosa
# import numpy as np
#
# root_local='/data/jfmadrig/amicorpus/5fold_valid_Silence/CV1/IS1004a/Closeup4/'
# files=os.listdir(root_local)
# wavfiles=[w for w in files if 'wav' in w]
# wavfiles=sorted(wavfiles)
# audio = None
# for wavfile in wavfiles:
# 	wav, sr = librosa.load(root_local+wavfile, sr=16000)
# 	if audio is None:
# 		audio = wav
# 	else:
# 		audio = np.append(audio,wav)
