import eval_toolkit
import os
import cv2
from pathlib import Path
import numpy as np

def load_data(class_names, video):
    with open(class_names, 'r') as f:
        class_names = f.readlines()
        f.close()
    # read GroudTruth
    gt_speakers, gt_faces = load_ground_truth(video)
    # read video
    cap = cv2.VideoCapture(video)
    return cap, class_names, gt_speakers, gt_faces

def load_ground_truth(video):
    if 'avdiar' in video:
        path = video.split('Video')
        gt_speakers, gt_faces = eval_toolkit.read_gt_avdiar(path[0])
        gt_speakers.insert(0, 'avdiar')
        return gt_speakers, gt_faces
    elif 'AVASM' in video:
        path = video.split('out.mp4')
        gt_speakers = eval_toolkit.read_gt_simple(path[0] + 'GT.txt')
        gt_speakers.insert(0, 'plain')
        return gt_speakers, []
    elif 'ibug-avs' in video:
        path = video.split('.')
        d = Path(video).resolve().parents[0]
        gt_file = str(d) + '/' + video.split('/')[-2] + '_00D.txt'
        gt_speakers = eval_toolkit.read_gt_simple(gt_file)
        gt_speakers.insert(0, 'plain')
        return gt_speakers, []
    else:
        return [],[]
    return [],[]


def split_speech_videos(videos_path, save_path_speaker, save_path_silence):

    subfolders = [f.path for f in os.scandir(videos_path) if f.is_dir()]
    subfolders.sort()
    for f in subfolders:
        video = f + '/' + f.split('/')[-1] + '_45D.mp4'
        process_video(video=video, save_path_speaker=save_path_speaker, save_path_silence=save_path_silence)
        video = f + '/' + f.split('/')[-1] + '_00D.mp4'
        process_video(video=video, save_path_speaker=save_path_speaker, save_path_silence=save_path_silence)
        video = f + '/' + f.split('/')[-1] + '_90D.mp4'
        process_video(video=video, save_path_speaker=save_path_speaker, save_path_silence=save_path_silence)


def split_videos_with_heads(videos_path, save_path_speaker, save_path_silence):

    subfolders = [f.path for f in os.scandir(videos_path) if f.is_dir()]
    subfolders.sort()
    for f in subfolders:
        video = f + '/' + f.split('/')[-1] + '_45D.mp4'
        process_video_heads(video=video, save_path_speaker=save_path_speaker, save_path_silence=save_path_silence)
        video = f + '/' + f.split('/')[-1] + '_00D.mp4'
        process_video_heads(video=video, save_path_speaker=save_path_speaker, save_path_silence=save_path_silence)
        video = f + '/' + f.split('/')[-1] + '_90D.mp4'
        process_video_heads(video=video, save_path_speaker=save_path_speaker, save_path_silence=save_path_silence)


def process_video_single(video,save_path):
    cap, class_names, gt_speakers, gt_faces = load_data('./lists/lrs2TrainTestlist/classInd.txt', video)

    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)

    output_video = None

    save_file = video.split('.')[0]
    save_file = save_file.split('/')[-1]
    save_file = save_path + '/' + save_file + '/'

    if not os.path.exists(save_file):
        os.makedirs(save_file)
    # else:
        # return

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    output_video = cv2.VideoWriter(save_file + 'output.avi',
                                   cv2.VideoWriter_fourcc(*'MP4V'),
                                   # cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   25,
                                   (fw, fh))
    for i in range(fps):
        ret, frame = cap.read()
        if not ret:
            break

        limit = min(i + 10, fps)
        speaking = False

        # if gt_speakers :
            # speaking = eval_toolkit.isSpeaking(gt_speakers, 0, frame=i)
        # if not sum(gt_speakers[i+1:limit]): # Silence
        if sum(gt_speakers[i+1:limit]) > 0:#== 0: # silence # > 0: # Voice
            cv2.imshow('result', frame)
            cv2.waitKey(1)
            output_video.write(frame)

    output_video.release()


def process_video(video, save_path_speaker, save_path_silence):
    cap, class_names, gt_speakers, gt_faces = load_data('./lists/lrs2TrainTestlist/classInd.txt', video)

    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.namedWindow('opticalflow', cv2.WINDOW_NORMAL)

    output_video_speaker = None
    output_video_silence = None

    save_file = video.split('.')[0]
    save_file = save_file.split('/')[-1]
    save_file_speaker = save_path_speaker + '/' + save_file + '/'
    save_file_silence = save_path_silence + '/' + save_file + '/'

    if not os.path.exists(save_file_speaker):
        os.makedirs(save_file_speaker)
    if not os.path.exists(save_file_silence):
        os.makedirs(save_file_silence)
    # else:
    # return

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    output_video_speaker = cv2.VideoWriter(save_file_speaker + 'output.avi',
                                           cv2.VideoWriter_fourcc(*'MP4V'),
                                           25,
                                           (fw, fh))
    output_video_silence = cv2.VideoWriter(save_file_silence + 'output.avi',
                                           cv2.VideoWriter_fourcc(*'MP4V'),
                                           25,
                                           (fw, fh))
    prvs, next, flow = [None, None, None]
    flow = np.zeros((fh, fw, 3), dtype=np.float32)

    for i in range(fps):
        ret, frame = cap.read()
        if not ret:
            break

        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prvs is not None:
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # np.save(image_file, flow)
            # flow = np.load(image_file + '.npy')
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            im_of = np.dstack((flow, mag))
            # im_of *= 16
            # im_of = cv2.normalize(im_of, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('opticalflow', bgr)
        else:
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255

        limit = min(i + 10, fps)

        if not sum(gt_speakers[i + 1:limit]):  # Silence
            cv2.imshow('result', frame)
            cv2.waitKey(1)
            output_video_silence.write(frame)
            np.save(save_file_silence + 'optflow', flow)
        elif sum(gt_speakers[i + 1:limit]) > 0:  # == 0: # silence # > 0: # Voice
            cv2.imshow('result', frame)
            cv2.waitKey(1)
            output_video_speaker.write(frame)
            np.save(save_file_speaker + 'optflow', flow)

        prvs = next

    output_video_speaker.release()
    output_video_silence.release()


def process_video_heads(video, save_path_speaker, save_path_silence):
    cap, class_names, gt_speakers, gt_faces = load_data('./lists/lrs2TrainTestlist/classInd.txt', video)

    from tools.linToDetector import linToDetector
    from tools.tracker_cv2 import cvTrackerAPI

    detector = linToDetector()
    tracker = cvTrackerAPI()

    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.namedWindow('head', cv2.WINDOW_NORMAL)
    cv2.namedWindow('opticalflow', cv2.WINDOW_NORMAL)

    output_video_speaker = None
    output_video_silence = None

    save_file = video.split('.')[0]
    save_file = save_file.split('/')[-1]
    save_file_speaker = save_path_speaker + '/' + save_file + '/'
    save_file_silence = save_path_silence + '/' + save_file + '/'
    save_file_speaker_head = (save_path_speaker[:-1] if save_path_speaker[-1] == '/' else save_path_speaker)+ '_head/' + save_file + '/'
    save_file_silence_head = (save_path_silence[:-1] if save_path_silence[-1] == '/' else save_path_silence)+ '_head/' + save_file + '/'

    if not os.path.exists(save_file_speaker):
        os.makedirs(save_file_speaker)
    if not os.path.exists(save_file_silence):
        os.makedirs(save_file_silence)

    if not os.path.exists(save_file_speaker_head):
        os.makedirs(save_file_speaker_head)
    if not os.path.exists(save_file_silence_head):
        os.makedirs(save_file_silence_head)
    # else:
    # return

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    output_video_speaker = cv2.VideoWriter(save_file_speaker + 'output.avi',
                                           cv2.VideoWriter_fourcc(*'MP4V'),
                                           25,
                                           (fw, fh))
    output_video_silence = cv2.VideoWriter(save_file_silence + 'output.avi',
                                           cv2.VideoWriter_fourcc(*'MP4V'),
                                           25,
                                           (fw, fh))
    prvs, next, flow = [None, None, None]
    flow = np.zeros((fh, fw, 3), dtype=np.float32)
    im_of_head = np.zeros((128, 171, 3), dtype=np.float32)

    fc_sil = 0
    fc_spk = 0

    for i in range(fps):
        ret, frame = cap.read()
        if not ret:
            break

        if (i % 2) == 0:
            detector.processFrame(frame, detFace=True, detShapes=False, regFace=False, poseEstim=False)
            tracker.processFrame(frame, detector.people_det[0], detector.faces_det[0])
        else:
            tracker.processFrame(frame)
            detector.clean()

            targets = tracker.get_box_p1p2()
            heads = tracker.get_box_head_p1p2()
            detector.processBoxes(frame, targets, heads, detFace=False, detShapes=False, regFace=False,
                                  poseEstim=False)

        if len(tracker.trackers) == 0 or tracker.trackers[0].box_head is None:
            continue

        x, y, w, h = tracker.trackers[0].box_head

        if w == 0 or h == 0:
            continue

        # x,y,w,h = detector.faces_det[0][0]
        x1 = int(max(x, 0))
        y1 = int(max(y, 0))
        x2 = int(min(x + w, fw))
        y2 = int(min(y + h, fh))
        head = frame[y1:y2, x1:x2]
        head = cv2.resize(head, (171, 128))
        cv2.imshow("head", head)


        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prvs is not None:
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # np.save(image_file, flow)
            # flow = np.load(image_file + '.npy')
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            im_of = np.dstack((flow, mag))
            im_of_head = im_of[y1:y2, x1:x2]
            im_of_head = cv2.resize(im_of_head, (171, 128))
            # im_of *= 16
            # im_of = cv2.normalize(im_of, None, 0, 255, cv2.NORM_MINMAX)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('opticalflow', bgr)
        else:
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255

        limit = min(i + 10, fps)

        if not sum(gt_speakers[i + 1:limit]):  # Silence
            output_video_silence.write(frame)
            save_name = save_file_silence_head + str(10000 + fc_sil) + '.jpg'
            cv2.imwrite(save_name, head)
            np.save(save_name, im_of_head)
            fc_sil += 1
        elif sum(gt_speakers[i + 1:limit]) > 0:  # == 0: # silence # > 0: # Voice
            output_video_speaker.write(frame)
            save_name = save_file_speaker_head + str(10000 + fc_spk) + '.jpg'
            cv2.imwrite(save_name, head)
            np.save(save_name, im_of_head)
            fc_spk += 1

        cv2.imshow('result', frame)
        cv2.waitKey(1)
        prvs = next

    output_video_speaker.release()
    output_video_silence.release()


def main():
    # split_speech_videos('/data/jfmadrig/ibug-avs/Digits/',
    #                     '/data/jfmadrig/ibug-avs/Voice_Digits_v2/',
    #                     '/data/jfmadrig/ibug-avs/Silence_Digits_v2/')
    split_videos_with_heads('/data/jfmadrig/ibug-avs/Digits/',
                            '/data/jfmadrig/ibug-avs/Voice_Digits_v2/',
                            '/data/jfmadrig/ibug-avs/Silence_Digits_v2/')
    # process_video('/data/jfmadrig/ibug-avs/Digits/S031_T01_L04_C01_R01/S031_T01_L04_C01_R01_90D.mp4', '/data/jfmadrig/ibug-avs/Voice_Digits/')

if __name__ == '__main__':
    main()
    exit()