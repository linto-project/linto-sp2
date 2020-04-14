import numpy as np
import cv2 as cv
import os
import glob


def process_images_in_subfolders(video_path = '/data/jfmadrig/hpedatasets/ict3DHP_heads/'):

    folders = sorted(os.listdir(video_path))

    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.namedWindow('frame op', cv.WINDOW_NORMAL)
    cv.namedWindow('frame new flow', cv.WINDOW_NORMAL)

    for folder in folders:
        if os.path.isfile(video_path + folder):
            continue

        image_list = sorted(os.listdir(video_path + '/' + folder))

        image_list = sorted(glob.glob(video_path + '/' + folder + '/*.jpg'))
        if image_list == []:
            image_list = sorted(glob.glob(video_path + '/' + folder + '/*.png'))
        print(image_list[0])
        frame = cv.imread(image_list[0], cv.IMREAD_COLOR)
        flow = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.float32)

        np.save(image_list[0], flow)
        # np.load(path + '.npy')

        prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255

        for image_file in image_list[1:]:
            frame = cv.imread(image_file, cv.IMREAD_COLOR)
            next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # np.save(image_file, flow)
            # flow = np.load(image_file + '.npy')
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            im_of = np.dstack((flow, mag))
            # im_of *= 16
            im_of = cv.normalize(im_of, None, 0, 255, cv.NORM_MINMAX)
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow('frame', frame)
            cv.imshow('frame op', bgr)
            cv.imshow('frame new flow', im_of)
            k = cv.waitKey(1) & 0xff
            if k == 27:
                break

            prvs = next


def process_vid():
    video_path = '/data/jfmadrig/VidTIMIT_heads/'

    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.namedWindow('frame op', cv.WINDOW_NORMAL)

    for (dirpath, dirnames, filenames) in os.walk(video_path):
        if len(filenames) == 0 or 'video' not in dirpath or 'head' not in dirpath:
            continue

        image_list = sorted([os.path.join(dirpath, file) for file in filenames])

        frame = cv.imread(image_list[0], cv.IMREAD_COLOR)
        flow = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.float32)

        np.save(image_list[0], flow)
        # np.load(path + '.npy')

        prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255

        for image_file in image_list[1:]:
            frame = cv.imread(image_file, cv.IMREAD_COLOR)
            next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            np.save(image_file, flow)
            flow = np.load(image_file + '.npy')
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow('frame', frame)
            cv.imshow('frame op', bgr)
            k = cv.waitKey(1) & 0xff
            if k == 27:
                break

            prvs = next


process_images_in_subfolders('/data/jfmadrig/ibug-avs/Silence_Digits_heads/')


def original():
    video_path = '/data/jfmadrig/hpedatasets/ict3DHP_heads/'
    # video_path = '/data/jfmadrig/VidTIMIT/fadg0/video/'
    # video_path = '/data/jfmadrig/VidTIMIT_heads/fadg0/video/'
    save_path = '/data/jfmadrig/hpedatasets/ict3DHP_heads_optflow/'

    folders = sorted(os.listdir(video_path))

    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    cv.namedWindow('frame op', cv.WINDOW_NORMAL)

    for folder in folders:
        if os.path.isfile(video_path + folder):
            continue

        image_list = sorted(os.listdir(video_path + '/' + folder))
        file = video_path + '/' + folder + '/' + image_list[0]
        frame = cv.imread(file,cv.IMREAD_COLOR)
        flow  = np.zeros((frame.shape[0], frame.shape[1], 2), dtype=np.float32)

        prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255

        for image_file in image_list[2:]:
            file = video_path + '/' + folder + '/' + image_file
            frame = cv.imread(file, cv.IMREAD_COLOR)
            next = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
            cv.imshow('frame',frame)
            cv.imshow('frame op',bgr)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv.imwrite('opticalfb.png',frame)
                cv.imwrite('opticalhsv.png',bgr)
            prvs = next


