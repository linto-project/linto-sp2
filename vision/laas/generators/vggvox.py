# Third Party
import librosa
import numpy as np
import random
from keras.utils import np_utils
import glob, os
# import generators.c3d as video

video_generator = []

# ===============================================
#       code from Arsha for loading data.
# ===============================================
def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


def preprocess(wav, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):

    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1] # inverse vector
    else:
        extended_wav = np.append(wav, wav[::-1])
    wav = extended_wav

    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    # if mode == 'train':
    #     randtime = np.random.randint(0, time-spec_len)
    #     spec_mag = mag_T[:, randtime:randtime+spec_len]
    # else:
    #     spec_mag = mag_T
    spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


def process_batch_audio(lines, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):

    num = len(lines)
    batch = np.zeros((num,257,129,1),dtype='float32')
    for i in range(num):
        path = lines[i].split(' ')[0]
        label = lines[i].split(' ')[-1]
        symbol = lines[i].split(' ')[1]
        label = label.strip('\n')
        label = int(label)
        symbol = int(symbol)-1
        # wavfiles = glob.glob(path+'/*.wav')
        # wavfiles.sort(key=str.lower)
        # if (symbol+16) >= len(wavfiles):
        #     continue
        if not os.path.isfile(path + '/{:06d}'.format(symbol + 16 ) + '.jpg'):
            continue

        wav = None

        for j in range(16):
            # wav_file = wavfiles[symbol + j]
            wav_file = path + '/{:06d}'.format(symbol + j) + '.wav'
            wav_short, sr = librosa.load(wav_file, sr=16000)

            if wav is None:
                wav = wav_short
            else:
                wav = np.append(wav_short, wav)

        batch[i,:,:,0] = preprocess(wav, win_length=win_length, sr=sr,
                                    hop_length=hop_length, n_fft=n_fft,
                                    spec_len=spec_len, mode=mode)
    return batch


def generator_train_batch_rgb_audio(train_txt,batch_size,num_classes,img_path):
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
            # x_train, x2_train, x_labels = siamese_process_batch_rgb(new_line[a:b],img_path,train=True)
            x_train, x_labels = video_generator.process_batch_rgb(new_line[a:b],img_path,train=True)
            x = video_generator.preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            x = np.transpose(x, (0,2,3,1,4))
            # x2= np.transpose(x2_train, (0,2,3,1,4))
            x2 = process_batch_audio(new_line[a:b], spec_len=129)
            # yield x, y
            yield [x, x2], y


def generator_val_batch_rgb_audio(val_txt,batch_size,num_classes,img_path):
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
            # y_test, y2_test, y_labels = siamese_process_batch_rgb(new_line[a:b],img_path,train=False)
            y_test, y_labels = video_generator.process_batch_rgb(new_line[a:b],img_path,train=False)
            x = video_generator.preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            # x2= np.transpose(y2_test, (0,2,3,1,4))
            x2 = process_batch_audio(new_line[a:b], spec_len=129, mode='test')
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield [x, x2], y


def generator_test_batch_rgb_audio(val_txt,batch_size,num_classes,img_path):
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
            # y_test, y2_test, y_labels = siamese_process_batch_rgb(new_line[a:b],img_path,train=False)
            y_test, y_labels = video_generator.process_batch_rgb(new_line[a:b],img_path,train=False)
            x = video_generator.preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            # x2= np.transpose(y2_test, (0,2,3,1,4))
            x2 = process_batch_audio(new_line[a:b], spec_len=129, mode='test')
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield [x, x2], y


def generator_train_batch_rgb_optflow_audio(train_txt,batch_size,num_classes,img_path):
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
            x_train, x2_train, x_labels = video_generator.siamese_process_batch(new_line[a:b],img_path,train=True)
            x = video_generator.preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            x = np.transpose(x, (0,2,3,1,4))
            x2= np.transpose(x2_train, (0,2,3,1,4))
            x3 = process_batch_audio(new_line[a:b], spec_len=129)
            # yield x, y
            yield [x, x2, x3], y


def generator_val_batch_rgb_optflow_audio(val_txt,batch_size,num_classes,img_path):
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
            y_test, y2_test, y_labels = video_generator.siamese_process_batch(new_line[a:b],img_path,train=False)
            x = video_generator.preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            x2= np.transpose(y2_test, (0,2,3,1,4))
            x3 = process_batch_audio(new_line[a:b], spec_len=129, mode='test')
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield [x, x2, x3], y


def generator_test_batch_rgb_optflow_audio(val_txt,batch_size,num_classes,img_path):
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
            y_test, y2_test, y_labels = video_generator.siamese_process_batch(new_line[a:b],img_path,train=False)
            x = video_generator.preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            x2= np.transpose(y2_test, (0,2,3,1,4))
            x3 = process_batch_audio(new_line[a:b], spec_len=129, mode='test')
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield [x, x2, x3], y
