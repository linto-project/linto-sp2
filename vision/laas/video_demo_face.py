# coding=utf8
from models.c3d import c3d_model
# import models.models as m
from keras.optimizers import SGD
import numpy as np
import os
import cv2
import statistics
import eval_toolkit
from tools.linToDetector import linToDetector
from tools.tracker_cv2 import cvTrackerAPI

detector = linToDetector()
tracker = cvTrackerAPI()
#
# kernel_w = 224
# kernel_h = 224
# img_w = 320 # 171
# img_h = 240 # 128
kernel_w = 112
kernel_h = 112
img_w = 171 # 171
img_h = 128 # 128
h_start = (img_h-kernel_h)//2
h_end   = h_start + kernel_h
w_start = (img_w-kernel_w)//2
w_end   = w_start + kernel_w

# detector = []
# tracker = []

num_classes = 2


def evaluate_faces(video, class_names, save_path=None, use_gt=False, tg_id=0):
    """
    Simple test
    :param video: Full path
    :param class_names: Label names
    :param save_path:
    :param use_gt:
    :param tg_id:
    :return:
    """
    save_file = None
    if save_path:

        save_file = video.split('.')[0]
        save_file = save_file.split('/')[-1]
        save_file = save_path + '/' + save_file + '_tg' + str(tg_id) + '/'

        if not os.path.exists(save_file):
            os.makedirs(save_file)
        else:
            return

    cap, class_names, model, gt_speakers, gt_faces = load_data(class_names, video)

    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow('result',cv2.WINDOW_NORMAL)
    cv2.namedWindow('head',cv2.WINDOW_NORMAL)

    TP, TN, FP, FN = [0,0,0,0]
    prob = []
    lbls = []
    gtls = []
    gtpb = []
    clip = []
    h = 0
    w = 0
    c = 0
    for i in range(fps):
        ret, frame = cap.read()
        if not ret:
            break

        h, w, c = frame.shape
        frame= cv2.resize(frame, (5*w, 5*h))

        h, w, c = frame.shape
        dim = [h//2, 0, h, w]

        head, pred_label, pred_prob = predict(model, frame, clip, class_names, dim)

        if gt_speakers and pred_label is not None:
            speaking = eval_toolkit.isSpeaking(gt_speakers, tg_id, i)
            gtls.append(1 if speaking else 0) # Set 1 if gt says speaking
            gtpb.append(pred_prob[0 if speaking else 1])

            # Add prob and true outcome
            lbls.append(1 if pred_label==0 else 0)
            prob.append(pred_prob[pred_label])
            if speaking and pred_label == 0:
                TP += 1
            elif speaking and pred_label == 1:
                FP += 1
            elif not speaking and pred_label == 0:
                FN += 1
            elif not speaking and pred_label == 1:
                TN += 1

        # if gt_speakers and pred_label is not None:
        #     comp = eval_results.isSpeaking(gt_speakers, pred_prob+1, i)
        #     if pred_prob == 0:
        #         if comp:
        #             TP += 1
        #         else:
        #             FN += 1
        #     else:
        #         if comp:
        #             FP += 1
        #         else:
        #             TN += 1
            # print(eval_results.isSpeaking(gt_speakers, label+1, i))
        cv2.imshow('result', frame)
        cv2.imshow('head', head)
        cv2.waitKey(1)

    # precision1 = TP / (TP+FP)
    # recall1 = TP/(TP+FN)

    ap, auc, f1, precision, recall = eval_toolkit.evaluate_performance(gtls, lbls, gtpb, prob, save_file)

    print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
    # print('Precision: ', statistics.mean(precision), 'Recall: ', statistics.mean(recall))


    cap.release()
    cv2.destroyAllWindows()


def simple_test(video, class_names, model_path, save_path=None, use_gt=False, tg_id=0, save_video=False):
    """
    Simple test
    :param video: Full path
    :param class_names: Label names
    :param save_path:
    :param use_gt:
    :param tg_id:
    :param save_video: Only works if save_path is set
    :return:
    """

    cap, class_names, model, gt_speakers, gt_faces = load_data(class_names, video, model_path)

    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow('result',cv2.WINDOW_NORMAL)
    cv2.namedWindow('head',cv2.WINDOW_NORMAL)

    save_file = None
    output_video = None
    if save_path:

        save_file = video.split('.')[0]
        save_file = save_file.split('/')[-1]
        save_file = save_path + '/' + save_file + '_tg' + str(tg_id) + '/'

        if not os.path.exists(save_file):
            os.makedirs(save_file)
        else:
            return

        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        if save_video:
            output_video = cv2.VideoWriter(save_file + 'output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (fw, fh))

    TP, TN, FP, FN = [0,0,0,0]
    prob = []
    lbls = []
    gtls = []
    gtpb = []
    clip = []
    h = 0
    w = 0
    f_count = 0
    for i in range(fps):
        ret, frame = cap.read()
        if not ret:
            break

        # frame = frame[:, :fw // 2]
        # frame = frame[:, fw // 2:]
        # frameorg = frame.copy()
        if use_gt:
            ok, dim = get_face(gt_faces, i, tg_id)
        else:
            ok, dim = detect_face(frame)
        if not ok:
            if clip:
                clip.pop(0)
            cv2.imshow('result', frame)
            cv2.waitKey(1)
            continue

        f_count += 1
        head, pred_label, pred_prob = predict(model, frame, clip, class_names, dim)

        if output_video:
            output_video.write(frame)

        if gt_speakers and pred_label is not None:
            speaking = eval_toolkit.isSpeaking(gt_speakers, tg_id, i)
            gtls.append(1 if speaking else 0) # Set 1 if gt says speaking
            gtpb.append(pred_prob[0 if speaking else 1])

            # Add prob and true outcome
            lbls.append(1 if pred_label==0 else 0)
            prob.append(pred_prob[pred_label])
            if speaking and pred_label == 0:
                TP += 1
            elif speaking and pred_label == 1:
                FP += 1
            elif not speaking and pred_label == 0:
                FN += 1
            elif not speaking and pred_label == 1:
                TN += 1

        # if gt_speakers and pred_label is not None:
        #     comp = eval_results.isSpeaking(gt_speakers, pred_prob+1, i)
        #     if pred_prob == 0:
        #         if comp:
        #             TP += 1
        #         else:
        #             FN += 1
        #     else:
        #         if comp:
        #             FP += 1
        #         else:
        #             TN += 1
            # print(eval_results.isSpeaking(gt_speakers, label+1, i))
        cv2.imshow('result', frame)
        cv2.imshow('head', head)
        cv2.waitKey(1)

    # precision1 = TP / (TP+FP)
    # recall1 = TP/(TP+FN)

    ap, auc, f1, precision, recall = eval_toolkit.evaluate_performance(gtls, lbls, gtpb, prob, save_file)
    mota = 1 - (FN+FP)/f_count
    if save_file:
        np.savetxt(save_file + '/mota.txt', [mota], fmt='%f')
    print('f1=%.3f auc=%.3f ap=%.3f mota:=%.3f' % (f1, auc, ap, mota))
    # print('Precision: ', statistics.mean(precision), 'Recall: ', statistics.mean(recall))


    cap.release()
    cv2.destroyAllWindows()


def detect_and_track_demo(video, class_names, model_path, save_path=None, use_gt=False, tg_id=0):
    """
    Simple test
    :param video: Full path
    :param class_names: Label names
    :param save_path:
    :param use_gt:
    :param tg_id:
    :return:
    """
    save_file = None
    if save_path:

        save_file = video.split('.')[0]
        save_file = save_file.split('/')[-1]
        save_file = save_path + '/' + save_file + '_tg' + str(tg_id) + '/'

        if not os.path.exists(save_file):
            os.makedirs(save_file)
        # else:
        #     return

    cap, class_names, model, gt_speakers, gt_faces = load_data(class_names, video, model_path)

    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow('result',cv2.WINDOW_NORMAL)
    cv2.namedWindow('head',cv2.WINDOW_NORMAL)

    TP, TN, FP, FN = [0,0,0,0]
    prob = []
    lbls = []
    gtls = []
    gtpb = []
    clip = []
    h = 0
    w = 0
    f_count = 0
    for i in range(fps):
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        # ret, frame = cap.read()
        ret, frame = cap.read()
        frame= cv2.resize(frame, (fw//2, fh//2))

        # img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
		#
        # # equalize the histogram of the Y channel
        # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
		#
        # # convert the YUV image back to RGB format
        # frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        if not ret:
            break

        # frame = frame[:, :fw // 2]
        # frame = frame[:, fw // 2:]
        # frameorg = frame.copy()
        if use_gt:
            ok, dim = get_face(gt_faces, i, tg_id)
        else:
            ok, dim = detect_and_track_face(frame, i)
        if not ok:
            cv2.imshow('result', frame)
            cv2.waitKey(1)
            continue

        f_count += 1
        head, pred_label, pred_prob = predict(model, frame, clip, class_names, dim)

        if gt_speakers and pred_label is not None:
            speaking = eval_toolkit.isSpeaking(gt_speakers, tg_id, i)
            gtls.append(1 if speaking else 0) # Set 1 if gt says speaking
            gtpb.append(pred_prob[0 if speaking else 1])

            # Add prob and true outcome
            lbls.append(1 if pred_label==0 else 0)
            prob.append(pred_prob[pred_label])
            if speaking and pred_label == 0:
                TP += 1
            elif speaking and pred_label == 1:
                FP += 1
            elif not speaking and pred_label == 0:
                FN += 1
            elif not speaking and pred_label == 1:
                TN += 1

        # if gt_speakers and pred_label is not None:
        #     comp = eval_results.isSpeaking(gt_speakers, pred_prob+1, i)
        #     if pred_prob == 0:
        #         if comp:
        #             TP += 1
        #         else:
        #             FN += 1
        #     else:
        #         if comp:
        #             FP += 1
        #         else:
        #             TN += 1
            # print(eval_results.isSpeaking(gt_speakers, label+1, i))
        cv2.imshow('result', frame)
        cv2.imshow('head', head)
        cv2.waitKey(1)

    # precision1 = TP / (TP+FP)
    # recall1 = TP/(TP+FN)

    ap, auc, f1, precision, recall = eval_toolkit.evaluate_performance(gtls, lbls, gtpb, prob, save_file)
    mota = 1 - (FN+FP)/f_count
    if save_file:
        np.savetxt(save_file + '/mota.txt', [mota], fmt='%f')

    print('f1=%.3f auc=%.3f ap=%.3f mota=%.3f' % (f1, auc, ap, mota))
    # print('Precision: ', statistics.mean(precision), 'Recall: ', statistics.mean(recall))


    cap.release()
    cv2.destroyAllWindows()


def get_face(gt_faces, iframe, tg_id=0):
    """
    Get the down part of the face
    :param gt_faces:
    :param iframe:
    :param tg_id:
    :return:
    """
    y1, x1, y2, x2 = [0, 0, 0, 0]

    ii = np.where(np.array(list(zip(*gt_faces))[0]) == iframe)[0]
    if ii.__len__() == 0:
        return False, [y1, x1, y2, x2]

    for i in ii:
        face = gt_faces[i]
        f, id, x1, y1, w, h = face[0], face[1], int(face[2]), int(face[3]), int(face[4]), int(face[5])
        if id == tg_id:
            x2 = x1 + w
            y2 = y1 + h
            y1 = y1 + h//2
            return True, [y1, x1, y2, x2]

    return False, [y1, x1, y2, x2]


def predict(model, frame, clip, class_names, dim):
    y1, x1, y2, x2 = dim
    head = frame[y1:y2, x1:x2]
    tmp = cv2.cvtColor(head, cv2.COLOR_BGR2RGB)
    clip.append(cv2.resize(tmp, (img_w, img_h)))
    label, pred = [None,None]
    if len(clip) == 16:
        inputs = np.array(clip).astype(np.float32)
        inputs = np.expand_dims(inputs, axis=0)
        inputs[..., 0] -= 99.9
        inputs[..., 1] -= 92.1
        inputs[..., 2] -= 82.6
        inputs[..., 0] /= 65.8
        inputs[..., 1] /= 62.3
        inputs[..., 2] /= 60.3
        # inputs = inputs[:, :, 8:120, 30:142, :]
        inputs = inputs[:, :, h_start:h_end, w_start:w_end, :]
        inputs = np.transpose(inputs, (0, 2, 3, 1, 4))
        pred = model.predict(inputs)
        label = np.argmax(pred[0])
        tracker.draw_trackers(frame)
        cv2.putText(frame, class_names[label][:-1], (20, 20),
                    # cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if label else (0, 0, 255), 2)
        cv2.putText(frame, "prob: %.4f" % pred[0][0], (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "prob: %.4f" % pred[0][1], (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("Speaker: %.4f Silence: %.4f" % (pred[0][0] , pred[0][1]) )
        if label == 0:
            m_color = (0, 0, 255)
        else:
            m_color = (0, 255, 0)

        cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (x1 + (x2 - x1) // 2, y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    m_color, 2)
        clip.pop(0)
        pred = pred[0]
    return head, label, pred


def load_data(class_names, video, model_path):
    with open(class_names, 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    model = c3d_model(num_classes, kernel_w, kernel_h)
    # model = m.r3d_34(num_classes, kernel_w, kernel_h)
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    # model.load_weights('./results/weights_c3d.h5', by_name=True)
    model.load_weights(model_path + '/weights_c3d.h5', by_name=True)
    # read GroudTruth
    gt_speakers, gt_faces = load_ground_truth(video)
    # read video
    cap = cv2.VideoCapture(video)
    return cap, class_names, model, gt_speakers, gt_faces


def detect_face(frame):
    detector.processFrame(frame, detFace=True, detShapes=False, regFace=False, poseEstim=False)
    y1, x1, y2, x2 = [0,0,0,0]
    if not detector.faces_det[0] or not detector.faces_det[0][0]:
        return False, [y1, x1, y2, x2]

    # detector.faces_det[0][0] = (img_h,186,190,255)
    # detector.faces_det[0][0] = (134,390,178,440)
    # detector.faces_det[0][0] = (79, 271, 150, 320)
    if detector.faces_det[1][0] > 0:
        (y1, x1, y2, x2) = detector.faces_det[0][0]
        # cy = y1 + (y2 - y1)//2
        # cx = x1 + (x2 - x1)//2
        #
        # if h == 0:
        h = (y2 - y1) // 2
        w = (x2 - x1) // 2
        y1 = y1 + h
    elif (x1 - x2) == 0:
        return False, [y1, x1, y2, x2]
    # x1 = int(max(cx - w, 0))
    # y1 = int(max(cy - h, 0))
    # x2 = int(min(cx + w, fw))
    # y2 = int(min(cy + h, fh))
    return True, [y1, x1, y2, x2]


def detect_and_track_face(frame, frame_id):
    # detector.processFrame(frame, detFace=True, detShapes=False, regFace=False, poseEstim=False)

    fh, fw, c = frame.shape
    y1, x1, y2, x2 = [0,0,0,0]

    if (frame_id % 10) == 0 or len(tracker.trackers) == 0 or tracker.trackers[0].box_head is None:
        detector.processFrame(frame, detFace=True, detShapes=False, regFace=False, poseEstim=False)
        if detector.faces_det[1] and detector.faces_det[1][0] > 0:
            tracker.processFrame(frame, detector.people_det[0], detector.faces_det[0])
    else:
        tracker.processFrame(frame)
        detector.clean()

        # targets = tracker.get_box_p1p2()
        # heads = tracker.get_box_head_p1p2()
        # detector.processBoxes(frame, targets, heads, detFace=True, detShapes=False, regFace=False, poseEstim=False)
        # tracker.processFrame(frame, detector.people_det[0], detector.faces_det[0])

    if len(tracker.trackers) == 0 or tracker.trackers[0].box_head is None:
        return False, [y1, x1, y2, x2]

    x, y, w, h = tracker.trackers[0].box_head

    # if not detector.faces_det[0] or not detector.faces_det[0][0]:
    if w == 0 or h == 0:
        return False, [y1, x1, y2, x2]

    x1 = int(max(x, 0))
    y1 = int(max(y + h//2, 0))
    x2 = int(min(x + w, fw))
    y2 = int(min(y + h, fh))

    return True, [y1, x1, y2, x2]


def load_ground_truth(video):
    if 'avdiar' in video:
        path = video.split('Video')
        gt_speakers, gt_faces = eval_toolkit.read_gt_avdiar(path[0])
        gt_speakers.insert(0, 'avdiar')
        return gt_speakers, gt_faces
    elif 'AVASM' in video:
        path = video.split('out.mp4')
        gt_speakers = eval_toolkit.read_gt_simple(path[0] + 'GT.txt')
        gt_speakers.insert(0, 'avasm')
        return gt_speakers, []
    elif 'ibug-avs' in video:
        path = video.split('.')
        gt_speakers = eval_toolkit.read_gt_simple(path[0] + '.txt')
        gt_speakers.insert(0, 'avasm')
        return gt_speakers, []
    else:
        return [],[]
    return [],[]


def resolution_test(video, class_names):
    with open(class_names, 'r') as f:
        class_names = f.readlines()
        f.close()

    # init model
    model = c3d_model(num_classes, kernel_w, kernel_h)
    # model = m.r3d_34(num_classes, kernel_w, kernel_h)
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
    model = c3d_model(num_classes, kernel_w, kernel_h)
    # model = m.r3d_34(num_classes, kernel_w, kernel_h)
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
            clip.append(cv2.resize(tmp, (img_w, img_h)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs[..., 0] -= 99.9
                inputs[..., 1] -= 92.1
                inputs[..., 2] -= 82.6
                inputs[..., 0] /= 65.8
                inputs[..., 1] /= 62.3
                inputs[..., 2] /= 60.3
                # inputs = inputs[:, :, 8:120, 30:142, :]
                inputs = inputs[:, :, h_start:h_end, w_start:w_end, :]
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
            clip.append(cv2.resize(tmp, (img_w, img_h)))
            if len(clip) == 16:
                inputs = np.array(clip).astype(np.float32)
                inputs = np.expand_dims(inputs, axis=0)
                inputs[..., 0] -= 99.9
                inputs[..., 1] -= 92.1
                inputs[..., 2] -= 82.6
                inputs[..., 0] /= 65.8
                inputs[..., 1] /= 62.3
                inputs[..., 2] /= 60.3
                # inputs = inputs[:, :, 8:120, 30:142, :]
                inputs = inputs[:, :, h_start:h_end, w_start:w_end, :]
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


def evaluate_avdiar():
    subfolders = [f.path for f in os.scandir('/data/jfmadrig/avdiar/') if f.is_dir()]
    subfolders.sort()
    # for f in subfolders:
    #     video = f + '/Video/' + f.split('/')[-1] + '_CAM1.mp4'
    #     targets = int(f.split('-')[1].split('P')[0])
    #     for i in range(1, targets + 1):
    #         simple_test(video, './lists/lrs2TrainTestlist/classInd.txt', './results_test/avdiar', True, i, save_video=True)

    results_path = './results_test/avdiar'
    get_all_results(results_path)


def evaluate_ibus_avs():

    subfolders = [f.path for f in os.scandir('/data/jfmadrig/ibug-avs/Phrases/') if f.is_dir()]
    subfolders.sort()
    # for f in subfolders:
    #     video = f + '/' + f.split('/')[-1] + '_00D.mp4'
    #     simple_test(video, './lists/lrs2TrainTestlist/classInd.txt', './results_test/ibug-avs/', save_video=True)

    results_path = './results_test/ibug-avs'
    get_all_results(results_path)


def get_all_results(results_path):
    gtls = []
    lbls = []
    gtpb = []
    prob = []
    mota = []
    folders = []
    for root, dirs, files in os.walk(results_path):
        for probFile in files:
            if not probFile.endswith("pred_prob.txt") or '/global' in root:
                continue

            lblsFile = root + '/gt_pred_labels.txt'
            gtls_local, lbls_local = np.loadtxt(lblsFile, comments="#", delimiter=" ", unpack=False)
            gtpb_local, prob_local = np.loadtxt(root + '/' + probFile, comments="#", delimiter=" ", unpack=False)
            mota_local = float(np.loadtxt(root + '/mota.txt', comments="#", delimiter=" ", unpack=False))

            gtls = np.concatenate((gtls, gtls_local), axis=None)
            lbls = np.concatenate((lbls, lbls_local), axis=None)
            gtpb = np.concatenate((gtpb, gtpb_local), axis=None)
            prob = np.concatenate((prob, prob_local), axis=None)
            mota = np.concatenate((mota, mota_local), axis=None)
            folders.append(root.split('/')[-1])

    ap, auc, f1, precision, recall = eval_toolkit.evaluate_performance(gtls, lbls, prob, gtpb, results_path + '/global/')
    np.savetxt(results_path + '/global/mota.txt', [statistics.mean(mota)], fmt='%f')

    tocheck = ['Seq01-1P-S0M1', 'Seq04-1P-S0M1', 'Seq22-1P-S0M1', 'Seq37-2P-S0M0', 'Seq43-2P-S0M0',
    'Seq37-2P-S0M0', 'Seq40-2P-S2M0', 'Seq20-2P-S1M1', 'Seq21-2P-S1M1', 'Seq44-2P-S2M0',
    'Seq12-3P-S1M1', 'Seq27-3P-S1M1', 'Seq13-4P-S2M1', 'Seq32-4P-S1M1']

    table = []
    for check in tocheck:
        table.append(0)
        count = 0
        for f in folders:
            if not check in f:
                continue
            table[-1] += mota[folders.index(f)]
            count += 1
        if count:
            table[-1] /= count

    np.savetxt(results_path + '/global/selected_mota.txt', [statistics.mean(table)], fmt = '%f')

    with open(results_path + '/global/table_mota.txt', 'w') as f:
        f.write("Mean : %f\n" % statistics.mean(mota))
        f.write("Mota : %f\n" )
        for item in mota:
            f.write("%f\n" % item)

def main():
    # simple_test('./videos/v_Biking_g05_c02.avi','./ucfTrainTestlist/classInd.txt')
    # simple_test('/data/jfmadrig/mvlrs_v1/pretrain/5672968469174139300/00004.mp4','./lrs2TrainTestlist/classInd.txt')
    # simple_test('/data/jfmadrig/VidTIMIT/fadg0/video/head/output.mp4','./lrs2TrainTestlist/classInd.txt')
    # simple_test('/data/jfmadrig/YouTubeFaces/frame_images_DB/Aaron_Eckhart/0/5.1404.avi', './lrs2TrainTestlist/classInd.txt')
    # simple_test('/data/jfmadrig/hpedatasets/ict3DHP_data/02/colour_undist.avi', './lrs2TrainTestlist/classInd.txt')
    # simple_test('/home/jfmadrig/Downloads/mcem0_head.mpg', './lrs2TrainTestlist/classInd.txt')

    # resolution_test('/data/jfmadrig/hpedatasets/ict3DHP_data/10/colour undist.avi', './lrs2TrainTestlist/classInd.txt')
    # ict3DHP_test('/data/jfmadrig/hpedatasets/ict3DHP_data/10/colour_undist.avi', './lists/lrs2TrainTestlist/classInd.txt')

    # simple_test('./videos/v_Biking_g05_c02.avi', './lrs2TrainTestlist/classInd.txt')

    # simple_test('/data/jfmadrig/TalkingFaceVideo/franck.mp4', './lists/lrs2TrainTestlist/classInd.txt', './results_test/TalkingFaceVideo/', save_video=True)
    # simple_test('/data/jfmadrig/TalkingFaceVideo/franck_00000.avi', './lists/lrs2TrainTestlist/classInd.txt')
    simple_test('/data/jfmadrig/hpedatasets/ict3DHP_data/10/colour_undist.avi', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/')
    # simple_test('/data/jfmadrig/YouTubeFaces/frame_images_DB/Ben_Curtis/0/0.avi', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/')
    # simple_test('/data/jfmadrig/YouTubeFaces/frame_images_DB/Carlos_Iturgaitz/5/out.mp4', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/')
    # simple_test('/home/jfmadrig/Downloads/mcem0_head.mpg', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/')
    # simple_test('/data/jfmadrig/mvlrs_v1/pretrain/5672968469174139300/00004.mp4', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/')
    # simple_test('/data/jfmadrig/avdiar/Seq43-2P-S0M0/Video/Seq43-2P-S0M0_CAM2.mp4', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/')
    # simple_test('/data/jfmadrig/avdiar/Seq40-2P-S2M0/Video/Seq40-2P-S2M0_CAM1.mp4', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/', './results_test/avdiar', True, i)
    # simple_test('/data/jfmadrig/ActivityNet/Crawler/Kinetics/videos/v_1RVu0qNtWCc.mp4', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/')
    # simple_test('/data/jfmadrig/amicorpus/ES2003a/video/ES2003a.Closeup1.avi', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/')
    # simple_test('/data/jfmadrig/RAVDESS/02-02-05-02-01-01-01.mp4', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/')
    # simple_test('/data/jfmadrig/RAVDESS/02-02-05-02-01-01-01.mp4', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/', './results_test/RAVDESS/', save_video=True)
    # simple_test('/data/jfmadrig/grid/id23_vcd_priazn.mpg', './lists/lrs2TrainTestlist/classInd.txt')

    # detect_and_track_demo('/data/jfmadrig/AVTRACK1/SS2/Data/video.avi', './lists/lrs2TrainTestlist/classInd.txt')
    # simple_test('/data/jfmadrig/YouTubeFaces/frame_images_DB/Barbara_Bodine/5/out.mp4', './lists/lrs2TrainTestlist/classInd.txt')
    # detect_and_track_demo('/data/jfmadrig/AVASM/moving-speaker/out.mp4', './lists/lrs2TrainTestlist/classInd.txt', './results_test/AVASM/')
    # detect_and_track_demo('/data/jfmadrig/modality/SPEAKER02_C1/SPEAKER02_C1_STRL.mp4', './lists/lrs2TrainTestlist/classInd.txt')
    # detect_and_track_demo('/data/jfmadrig/maxplant/JEVM392_au51-54_CamC.mpg', './lists/lrs2TrainTestlist/classInd.txt')
    detect_and_track_demo('/data/jfmadrig/ibug-avs/Phrases/S003_T02_L04_C01_R01/S003_T02_L04_C01_R01_00D.mp4', './lists/lrs2TrainTestlist/classInd.txt', model_path='./weights/C2E20-lr-vid-ict_lips_randomScale/')
    # simple_test('/data/jfmadrig/ibug-avs/Phrases/S015_T02_L04_C01_R01/S015_T02_L04_C01_R01_00D.mp4', './lists/lrs2TrainTestlist/classInd.txt', './results_test/ibug-avs/', save_video=True)

    # evaluate_ibus_avs()

    # evaluate_avdiar()

    # evaluate_faces('/data/jfmadrig/cuave/video1/g01_aligned.avi', './lists/lrs2TrainTestlist/classInd.txt')

    # results_path = './results_test/avdiar'
    # get_all_results(results_path)


if __name__ == '__main__':
    main()
    exit()