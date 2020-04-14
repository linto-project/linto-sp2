# coding=utf8
from models import c3d as c3d_model
from keras.optimizers import SGD
import numpy as np
import cv2

num_classes = 2

def simple_test(video, class_names):
    with open(class_names, 'r') as f:
        class_names = f.readlines()
        f.close()

    # init model
    model = c3d_model(num_classes)
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    model.load_weights('./results/weights_c3d.h5', by_name=True)

    # read video
    # video = './videos/v_Biking_g05_c02.avi'
    # video = '/data/jfmadrig/mvlrs_v1/pretrain/5672968469174139300/00004.mp4'
    cap = cv2.VideoCapture(video)

    clip = []
    while True:
        ret, frame = cap.read()
        if ret:
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (171, 128)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs[..., 0] -= 99.9
                inputs[..., 1] -= 92.1
                inputs[..., 2] -= 82.6
                inputs[..., 0] /= 65.8
                inputs[..., 1] /= 62.3
                inputs[..., 2] /= 60.3
                inputs = inputs[:,:,8:120,30:142,:]
                inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
                pred = model.predict(inputs)
                label = np.argmax(pred[0])
                cv2.putText(frame, class_names[label][:-1], (20, 20),
                # cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                cv2.putText(frame, "prob: %.4f" % pred[0][label], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                clip.pop(0)
            cv2.imshow('result', frame)
            cv2.waitKey(10)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def resolution_test(video, class_names):
    with open(class_names, 'r') as f:
        class_names = f.readlines()
        f.close()

    # init model
    model = c3d_model(num_classes)
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    model.load_weights('./results/weights_c3d.h5', by_name=True)

    # read video
    # video = './videos/v_Biking_g05_c02.avi'
    # video = '/data/jfmadrig/mvlrs_v1/pretrain/5672968469174139300/00004.mp4'
    cap = cv2.VideoCapture(video)
    gt_index = 1
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    with open('result_video_demo.txt', 'w') as fp:
        fp.write('Resolution\tFP\tMean FP\tTP\tMean TP\n')

        for i in range(1, 10):
            tmp_w = int(w / i)
            tmp_h = int(h / i)
            data = process_video(cap, class_names, model, gt_index,(tmp_w, tmp_h) )
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                (tmp_w, tmp_h), data[1], data[2], data[3], data[4]))

        fp.close()
    cap.release()
    cv2.destroyAllWindows()

def ict3DHP_test(video, class_names):
    with open(class_names, 'r') as f:
        class_names = f.readlines()
        f.close()

    # init model
    model = c3d_model(num_classes)
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    model.load_weights('./results/weights_c3d.h5', by_name=True)

    # read video
    # video = './videos/v_Biking_g05_c02.avi'
    # video = '/data/jfmadrig/mvlrs_v1/pretrain/5672968469174139300/00004.mp4'
    cap = cv2.VideoCapture(video)
    gt_index = 1
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    with open('result_video_demo.txt', 'w') as fp:
        fp.write('Resolution\tFP\tMean FP\tTP\tMean TP\n')

        for i in range(1, 10):
            tmp_w = int(w / i)
            tmp_h = int(h / i)
            data = process_video_ict3DHP(cap, class_names, model, gt_index,(tmp_w, tmp_h) )
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                (tmp_w, tmp_h), data[1], data[2], data[3], data[4]))

        fp.close()
    cap.release()
    cv2.destroyAllWindows()

def process_video_ict3DHP(cap, class_names, model, gt_idx, new_size = None):
    clip = []
    fp = 0
    tp = 0
    mn_tp = 0
    mn_fp = 0
    n_f = 0

    while True:
        ret, frame = cap.read()
        if ret:
            # if new_size is not None:
                # frame = cv2.resize(frame, new_size)
            frame = frame[200:300, 255:395]
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (171, 128)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs[..., 0] -= 99.9
                inputs[..., 1] -= 92.1
                inputs[..., 2] -= 82.6
                inputs[..., 0] /= 65.8
                inputs[..., 1] /= 62.3
                inputs[..., 2] /= 60.3
                inputs = inputs[:, :, 8:120, 30:142, :]
                inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
                pred = model.predict(inputs)
                label = np.argmax(pred[0])
                cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                cv2.putText(frame, "prob: %.4f" % pred[0][label], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                clip.pop(0)
                n_f += 1
                if gt_idx != label:
                    fp += 1
                    mn_fp += pred[0][label]
                else:
                    tp += 1
                    mn_tp += pred[0][label]


            cv2.imshow('result', frame)
            cv2.waitKey(10)
        else:
            break

    if fp != 0:
        mn_fp /= fp
    if tp != 0:
        mn_tp /= tp

    return  [n_f, fp, mn_fp, tp, mn_tp]


def process_video(cap, class_names, model, gt_idx, new_size = None):
    clip = []
    fp = 0
    tp = 0
    mn_tp = 0
    mn_fp = 0
    n_f = 0

    while True:
        ret, frame = cap.read()
        if ret:
            if new_size is not None:
                frame = cv2.resize(frame, new_size)
            tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip.append(cv2.resize(tmp, (171, 128)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs[..., 0] -= 99.9
                inputs[..., 1] -= 92.1
                inputs[..., 2] -= 82.6
                inputs[..., 0] /= 65.8
                inputs[..., 1] /= 62.3
                inputs[..., 2] /= 60.3
                inputs = inputs[:, :, 8:120, 30:142, :]
                inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
                pred = model.predict(inputs)
                label = np.argmax(pred[0])
                cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                cv2.putText(frame, "prob: %.4f" % pred[0][label], (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 1)
                clip.pop(0)
                n_f += 1
                if gt_idx != label:
                    fp += 1
                    mn_fp += pred[0][label]
                else:
                    tp += 1
                    mn_tp += pred[0][label]


            cv2.imshow('result', frame)
            cv2.waitKey(10)
        else:
            break

    if fp != 0:
        mn_fp /= fp
    if tp != 0:
        mn_tp /= tp

    return  [n_f, fp, mn_fp, tp, mn_tp]


def main():
    # simple_test('./videos/v_Biking_g05_c02.avi','./ucfTrainTestlist/classInd.txt')
    # simple_test('/data/jfmadrig/mvlrs_v1/pretrain/5672968469174139300/00004.mp4','./lrs2TrainTestlist/classInd.txt')
    # simple_test('/data/jfmadrig/VidTIMIT/fadg0/video/head/output.mp4','./lrs2TrainTestlist/classInd.txt')
    # simple_test('/data/jfmadrig/YouTubeFaces/frame_images_DB/Aaron_Eckhart/0/5.1404.avi', './lrs2TrainTestlist/classInd.txt')
    # simple_test('/data/jfmadrig/hpedatasets/ict3DHP_data/02/colour_undist.avi', './lrs2TrainTestlist/classInd.txt')
    # simple_test('/home/jfmadrig/Downloads/mcem0_head.mpg', './lrs2TrainTestlist/classInd.txt')

    # resolution_test('/data/jfmadrig/hpedatasets/ict3DHP_data/10/colour undist.avi', './lrs2TrainTestlist/classInd.txt')
    # ict3DHP_test('/data/jfmadrig/hpedatasets/ict3DHP_data/10/colour_undist.avi', './lrs2TrainTestlist/classInd.txt')

    # simple_test('./videos/v_Biking_g05_c02.avi', './lrs2TrainTestlist/classInd.txt')
    simple_test('/data/jfmadrig/TalkingFaceVideo/franck_00000.avi', './lrs2TrainTestlist/classInd.txt')

if __name__ == '__main__':
    main()