import cv2
import os

from tools.linToDetector import linToDetector
from tools.tracker_cv2 import cvTrackerAPI

detector = linToDetector()
tracker = cvTrackerAPI()


def ucf_dataset():
    video_path = '/local/users/jfmadrig/data/UCF-101/'
    save_path = '/local/users/jfmadrig/data/ucfimgs/'

    action_list = os.listdir(video_path)

    for action in action_list:
        if not os.path.exists(save_path+action):
            os.mkdir(save_path+action)
        video_list = os.listdir(video_path+action)
        for video in video_list:
            prefix = video.split('.')[0]
            if not os.path.exists(save_path+action+'/'+prefix):
                os.mkdir(save_path+action+'/'+prefix)
            save_name = save_path + action + '/' + prefix + '/'
            video_name = video_path+action+'/'+video
            cap = cv2.VideoCapture(video_name)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_count = 0
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(save_name+str(10000+fps_count)+'.jpg', frame)
                    fps_count += 1


def mvlrs_v1_dataset():
    video_path = '/data/jfmadrig/mvlrs_v1/pretrain/'
    save_path = '/data/jfmadrig/mvlrs_v1/pretrain_heads/'

    action_list = sorted(os.listdir(video_path))
    limit = 1000
    # To recover for last crash
    # toCont = True

    for indx, action in enumerate(action_list):
        # To recover for last crash
        if indx < 53:
            continue
        if not os.path.exists(save_path + action):
            os.mkdir(save_path + action)
        video_list = os.listdir(video_path + action)
        for video in video_list:
            # # To recover for last crash
            # if indx == 53 and toCont and video != '00028.mp4':
            #     continue
            # elif indx == 53 and video == '00028.mp4':
            #     toCont = False

            prefix, suffix = video.split('.')
            if suffix == 'txt':
                continue
            if not os.path.exists(save_path + action + '/' + prefix):
                os.mkdir(save_path + action + '/' + prefix)
            save_name = save_path + action + '/' + prefix + '/'
            video_name = video_path + action + '/' + video
            cap = cv2.VideoCapture(video_name)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_count = 0
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    # Detect all elements (person, face, gesture)
                    if (fps_count % 2) == 0:
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

                    x,y,w,h = tracker.trackers[0].box_head

                    if w == 0 or h == 0:
                        continue

                    # x,y,w,h = detector.faces_det[0][0]
                    x1 = int(max(x,0))
                    y1 = int(max(y,0))
                    x2 = int(min(x+w,fw))
                    y2 = int(min(y+h,fh))
                    head = frame[y1:y2, x1:x2]
                    head = cv2.resize(head, (171, 128))
                    cv2.imshow("head", head)


                    cv2.imwrite(save_name + str(10000 + fps_count) + '.jpg', head)
                    fps_count += 1

                    tracker.draw_trackers(frame)
                    cv2.imshow("preview", frame)
                    key = cv2.waitKey(1)

            tracker.clean()
            detector.clean()

        print(video_path + action + " Done")

        if indx == limit :
            break


def vid_dataset():
    video_path = '/data/jfmadrig/VidTIMIT/'
    save_path = '/data/jfmadrig/VidTIMIT_heads/'
    count = 0
    for (dirpath, dirnames, filenames) in os.walk(video_path):
        if len(filenames) == 0 or 'video' not in dirpath or 'head' not in dirpath:
            continue

        # count += 1
        # if count < 75:
        #     continue

        listOfImages = [os.path.join(dirpath, file) for file in filenames]
        # listOfHeads  = [image.replace(video_path, save_path) for image in listOfImages]
        # listOfFiles = [os.path.join(dirpath, file) for file in filenames if '.jpg' in file ]

        listOfImages.sort(key=str.lower)
        # listOfHeads.sort(key=str.lower)

        save_name = dirpath.replace(video_path, save_path)

        if not os.path.exists(save_name):
            os.makedirs(save_name)
        else:
            continue

        fps = len(listOfImages)
        fps_count = 0
        y1 = 0
        x1 = 0
        y2 = 0
        x2 = 0
        for i, image in enumerate(listOfImages):
            frame = cv2.imread(image)
            if frame is None:
                continue

            fw, fh, c = frame.shape

            # Detect all elements (person, face, gesture)
            if (fps_count % 1) == 0:
                detector.processFrame(frame, detFace=True, detShapes=False, regFace=False, poseEstim=False)
                # tracker.processFrame(frame, detector.people_det[0], detector.faces_det[0])
                # tracker.processFrame(frame, detector.people_det[0])
            else:
                tracker.processFrame(frame)
                detector.clean()

                targets = tracker.get_box_p1p2()
                heads = tracker.get_box_head_p1p2()
                detector.processBoxes(frame, targets, heads, detFace=False, detShapes=False, regFace=False,
                                      poseEstim=False)

            # if len(tracker.trackers) == 0 or tracker.trackers[0].box_head is None:
            #     continue
            #
            # x, y, w, h = tracker.trackers[0].box_head
            #
            # if w == 0 or h == 0:
            #     continue
            #
            # # x,y,w,h = detector.faces_det[0][0]
            # x1 = int(max(x, 0))
            # y1 = int(max(y, 0))
            # x2 = int(min(x + w, fw))
            # y2 = int(min(y + h, fh))

            if not detector.faces_det[1]:
                continue

            if detector.faces_det[1][0] > 0:
                (y1, x1, y2, x2) = detector.faces_det[0][0]
            elif (x1-x2) == 0:
                continue

            head = frame[y1:y2, x1:x2]
            head = cv2.resize(head, (171, 128))
            cv2.imshow("head", head)

            cv2.imwrite(save_name + '/' + str(10000 + fps_count) + '.jpg', head)
            fps_count += 1

            # tracker.draw_trackers(frame)
            cv2.imshow("preview", frame)
            key = cv2.waitKey(1)

        tracker.clean()
        detector.clean()

    print(dirpath + " Done")


def vid_dataset_slow():
    video_path = '/data/jfmadrig/VidTIMIT/'
    save_path = '/data/jfmadrig/VidTIMIT_heads_slow/'
    count = 0
    for (dirpath, dirnames, filenames) in os.walk(video_path):
        if len(filenames) == 0 or 'video' not in dirpath or 'head' not in dirpath:
            continue

        # count += 1
        # if count < 75:
        #     continue

        listOfImages = [os.path.join(dirpath, file) for file in filenames]
        # listOfHeads  = [image.replace(video_path, save_path) for image in listOfImages]
        # listOfFiles = [os.path.join(dirpath, file) for file in filenames if '.jpg' in file ]

        listOfImages.sort(key=str.lower)
        # listOfHeads.sort(key=str.lower)

        save_name = dirpath.replace(video_path, save_path)

        if not os.path.exists(save_name):
            os.makedirs(save_name)
        else:
            continue

        fps = len(listOfImages)
        fps_count = 0
        y1 = 0
        x1 = 0
        y2 = 0
        x2 = 0
        for i, image in enumerate(listOfImages):
            frame = cv2.imread(image)
            if frame is None:
                continue

            fw, fh, c = frame.shape

            # Detect all elements (person, face, gesture)
            if (fps_count % 1) == 0:
                detector.processFrame(frame, detFace=True, detShapes=False, regFace=False, poseEstim=False)
                # tracker.processFrame(frame, detector.people_det[0], detector.faces_det[0])
                # tracker.processFrame(frame, detector.people_det[0])
            else:
                tracker.processFrame(frame)
                detector.clean()

                targets = tracker.get_box_p1p2()
                heads = tracker.get_box_head_p1p2()
                detector.processBoxes(frame, targets, heads, detFace=False, detShapes=False, regFace=False,
                                      poseEstim=False)

            # if len(tracker.trackers) == 0 or tracker.trackers[0].box_head is None:
            #     continue
            #
            # x, y, w, h = tracker.trackers[0].box_head
            #
            # if w == 0 or h == 0:
            #     continue
            #
            # # x,y,w,h = detector.faces_det[0][0]
            # x1 = int(max(x, 0))
            # y1 = int(max(y, 0))
            # x2 = int(min(x + w, fw))
            # y2 = int(min(y + h, fh))

            if not detector.faces_det[1]:
                continue

            if detector.faces_det[1][0] > 0:
                (y1, x1, y2, x2) = detector.faces_det[0][0]
            elif (x1 - x2) == 0:
                continue

            head = frame[y1:y2, x1:x2]
            head = cv2.resize(head, (171, 128))
            cv2.imshow("head", head)

            for i in range(16):
                cv2.imwrite(save_name + '/' + str(10000 + fps_count) + '.jpg', head)
                fps_count += 1

            # tracker.draw_trackers(frame)
            cv2.imshow("preview", frame)
            key = cv2.waitKey(1)

        tracker.clean()
        detector.clean()

        print(dirpath + " Done")


def afew_dataset():
    video_path = '/data/jfmadrig/AFEW-VA/'
    save_path = '/data/jfmadrig/AFEW-VA_heads/'

    for (dirpath, dirnames, filenames) in os.walk(video_path):


        listOfImages = [os.path.join(dirpath, file) for file in filenames]
        listOfImages.sort(key=str.lower)

        save_name = dirpath.replace(video_path, save_path)

        if not os.path.exists(save_name):
            os.makedirs(save_name)
        else:
            continue

        fps = len(listOfImages)
        fps_count = 0
        y1 = 0
        x1 = 0
        y2 = 0
        x2 = 0
        for i, image in enumerate(listOfImages):
            frame = cv2.imread(image)
            if frame is None:
                continue

            fw, fh, c = frame.shape

            # Detect all elements (person, face, gesture)
            detector.processFrame(frame, detFace=True, detShapes=False, regFace=False, poseEstim=False)

            if not detector.faces_det[1]:
                continue

            if detector.faces_det[1][0] > 0:
                (y1, x1, y2, x2) = detector.faces_det[0][0]
            elif (x1-x2) == 0:
                continue

            head = frame[y1:y2, x1:x2]
            head = cv2.resize(head, (171, 128))
            cv2.imshow("head", head)

            cv2.imwrite(save_name + '/' + str(10000 + fps_count) + '.jpg', head)
            fps_count += 1

            # tracker.draw_trackers(frame)
            cv2.imshow("preview", frame)
            key = cv2.waitKey(1)

        tracker.clean()
        detector.clean()

    print(dirpath + " Done")


def ict3DHP():
    video_path = '/data/jfmadrig/hpedatasets/ict3DHP_data/'
    save_path  = '/data/jfmadrig/hpedatasets/ict3DHP_heads/'

    action_list = sorted(os.listdir(video_path))
    limit = 1000
    # To recover for last crash
    # toCont = True

    for indx, action in enumerate(action_list):
        # To recover for last crash
        if indx < 3:
            continue
        if os.path.isfile(video_path + action):
            continue
        if not os.path.exists(save_path + action):
            os.makedirs(save_path + action)
        video_list = os.listdir(video_path + action)
        for video in video_list:
            # # To recover for last crash
            # if indx == 53 and toCont and video != '00028.mp4':
            #     continue
            # elif indx == 53 and video == '00028.mp4':
            #     toCont = False
            if os.path.isdir(video_path + action + '/' + video):
                continue

            prefix, suffix = video.split('.')
            if suffix != 'avi':
                continue

            save_name = save_path + action + '/'
            video_name = video_path + action + '/' + video
            cap = cv2.VideoCapture(video_name)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_count = 0
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    # Detect all elements (person, face, gesture)
                    if (fps_count % 2) == 0:
                        detector.processFrame(frame, detFace=True, detShapes=False, regFace=False, poseEstim=False)
                        tracker.processFrame(frame, detector.people_det[0], detector.faces_det[0])
                    else:
                        tracker.processFrame(frame)
                        detector.clean()

                        targets = tracker.get_box_p1p2()
                        heads = tracker.get_box_head_p1p2()
                        detector.processBoxes(frame, targets, heads, detFace=False, detShapes=False, regFace=False,
                                              poseEstim=False)

                    if len(tracker.trackers) == 0  or tracker.trackers[0].box_head is None:
                        continue

                    x,y,w,h = tracker.trackers[0].box_head

                    if w == 0 or h == 0:
                        continue

                    # x,y,w,h = detector.faces_det[0][0]
                    x1 = int(max(x,0))
                    y1 = int(max(y,0))
                    x2 = int(min(x+w,fw))
                    y2 = int(min(y+h,fh))
                    head = frame[y1:y2, x1:x2]
                    head = cv2.resize(head, (171, 128))
                    cv2.imshow("head", head)


                    cv2.imwrite(save_name + str(10000 + fps_count) + '.jpg', head)
                    fps_count += 1

                    tracker.draw_trackers(frame)
                    cv2.imshow("preview", frame)
                    key = cv2.waitKey(1)

            tracker.clean()
            detector.clean()

        print(video_path + action + " Done")

        if indx == limit :
            break


def get_heads_from_video(video_path, save_path):

    action_list = sorted(os.listdir(video_path))
    # To recover for last crash
    # toCont = True

    for indx, action in enumerate(action_list):
        # To recover for last crash
        # if indx < 3:
        #     continue
        if os.path.isfile(video_path + action):
            continue
        if not os.path.exists(save_path + action):
            os.makedirs(save_path + action)
        else:
            continue
        video_list = os.listdir(video_path + action)
        for video in video_list:
            # # To recover for last crash
            # if indx == 53 and toCont and video != '00028.mp4':
            #     continue
            # elif indx == 53 and video == '00028.mp4':
            #     toCont = False
            if os.path.isdir(video_path + action + '/' + video):
                continue

            prefix, suffix = video.split('.')
            if suffix != 'avi':
                continue

            save_name = save_path + action + '/'
            video_name = video_path + action + '/' + video
            cap = cv2.VideoCapture(video_name)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_count = 0
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    # Detect all elements (person, face, gesture)
                    if (fps_count % 2) == 0:
                        detector.processFrame(frame, detFace=True, detShapes=False, regFace=False, poseEstim=False)
                        tracker.processFrame(frame, detector.people_det[0], detector.faces_det[0])
                    else:
                        tracker.processFrame(frame)
                        detector.clean()

                        targets = tracker.get_box_p1p2()
                        heads = tracker.get_box_head_p1p2()
                        detector.processBoxes(frame, targets, heads, detFace=False, detShapes=False, regFace=False,
                                              poseEstim=False)

                    if len(tracker.trackers) == 0  or tracker.trackers[0].box_head is None:
                        continue

                    x,y,w,h = tracker.trackers[0].box_head

                    if w == 0 or h == 0:
                        continue

                    # x,y,w,h = detector.faces_det[0][0]
                    x1 = int(max(x,0))
                    y1 = int(max(y,0))
                    x2 = int(min(x+w,fw))
                    y2 = int(min(y+h,fh))
                    head = frame[y1:y2, x1:x2]
                    head = cv2.resize(head, (171, 128))
                    cv2.imshow("head", head)


                    cv2.imwrite(save_name + str(10000 + fps_count) + '.jpg', head)
                    fps_count += 1

                    tracker.draw_trackers(frame)
                    cv2.imshow("preview", frame)
                    key = cv2.waitKey(1)

            tracker.clean()
            detector.clean()

        print(video_path + action + " Done")


def get_static_heads_from_video(video_path, save_path):

    action_list = sorted(os.listdir(video_path))
    # To recover for last crash
    # toCont = True

    for indx, action in enumerate(action_list):
        # To recover for last crash
        # if indx < 3:
        #     continue
        if os.path.isfile(video_path + action):
            continue
        video_list = os.listdir(video_path + action)
        for video in video_list:
            # # To recover for last crash
            # if indx == 53 and toCont and video != '00028.mp4':
            #     continue
            # elif indx == 53 and video == '00028.mp4':
            #     toCont = False
            if os.path.isdir(video_path + action + '/' + video):
                continue

            prefix, suffix = video.split('.')
            if suffix != 'avi' and suffix != 'mp4':
                continue

            save_name = save_path + prefix + '/'

            if not os.path.exists(save_name):
                os.makedirs(save_name)
            else:
                continue

            video_name = video_path + action + '/' + video
            cap = cv2.VideoCapture(video_name)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_count = 0
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    # Detect all elements (person, face, gesture)
                    if fps_count  == 0 or len(tracker.trackers) == 0  or tracker.trackers[0].box_head is None:
                        detector.processFrame(frame, detFace=True, detShapes=False, regFace=False, poseEstim=False)
                        tracker.processFrame(frame, detector.people_det[0], detector.faces_det[0])

                    if len(tracker.trackers) == 0  or tracker.trackers[0].box_head is None:
                        continue

                    x,y,w,h = tracker.trackers[0].box_head

                    if w == 0 or h == 0:
                        continue

                    # x,y,w,h = detector.faces_det[0][0]
                    x1 = int(max(x,0))
                    y1 = int(max(y,0))
                    x2 = int(min(x+w,fw))
                    y2 = int(min(y+h,fh))
                    head = frame[y1:y2, x1:x2]
                    head = cv2.resize(head, (171, 128))
                    cv2.imshow("head", head)


                    cv2.imwrite(save_name + str(10000 + fps_count) + '.jpg', head)
                    fps_count += 1

                    tracker.draw_trackers(frame)
                    cv2.imshow("preview", frame)
                    key = cv2.waitKey(1)

            tracker.clean()
            detector.clean()

        print(video_path + action + " Done")


def get_lips(in_path, out_path):
    # in_path = '/data/jfmadrig/AFEW-VA/'
    # out_path = '/data/jfmadrig/AFEW-VA_heads/'

    for (dirpath, dirnames, filenames) in os.walk(in_path):

        if len(filenames) == 0:
            continue

        listOfImages = [os.path.join(dirpath, file) for file in filenames]
        listOfImages.sort(key=str.lower)

        save_name = dirpath.replace(in_path, out_path)

        if not os.path.exists(save_name):
            os.makedirs(save_name)
        else:
            continue

        fps_count = 0
        for i, image in enumerate(listOfImages):
            frame = cv2.imread(image)
            if frame is None:
                continue

            fw, fh, c = frame.shape
            lips = frame[fh//2-10:fh-10, :]

            cv2.imshow("lips", lips)

            cv2.imwrite(save_name + '/' + str(10000 + fps_count) + '.jpg', lips)
            fps_count += 1

            cv2.imshow("preview", frame)
            key = cv2.waitKey(1)

    print(dirpath + " Done")




if __name__ == '__main__':
    # ucf_dataset()
    # mvlrs_v1_dataset()
    # vid_dataset()
    # afew_dataset()
    # vid_dataset_slow()
    # ict3DHP()
    # get_lips('/data/jfmadrig/mvlrs_v1/pretrain_heads/', '/data/jfmadrig/mvlrs_v1/pretrain_lips/')
    # get_lips('/data/jfmadrig/VidTIMIT_heads/', '/data/jfmadrig/VidTIMIT_lips/')
    get_heads_from_video('/data/jfmadrig/ibug-avs/Silence_Digits/', '/data/jfmadrig/ibug-avs/Silence_Digits_heads/')
    # get_lips('/data/jfmadrig/ibug-avs/Voice_Digits_heads/', '/data/jfmadrig/ibug-avs/Voice_Digits_heads_lips/')

    # get_static_heads_from_video('/data/jfmadrig/ibug-avs/Digits/', '/data/jfmadrig/ibug-avs/Static_Digits_heads/')
    # get_lips('/data/jfmadrig/ibug-avs/Static_Digits_heads/', '/data/jfmadrig/ibug-avs/Static_Digits_lips/')