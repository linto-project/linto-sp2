import cv2
import os


def get_lips(in_path, out_path, step = 1):
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
            if i % step != 0:
                continue
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
    # get_lips('/data/jfmadrig/mvlrs_v1/pretrain_heads/', '/data/jfmadrig/mvlrs_v1/pretrain_lips/')
    # get_lips('/data/jfmadrig/VidTIMIT_heads/', '/data/jfmadrig/VidTIMIT_lips/')
    # get_lips('/data/jfmadrig/VidTIMIT_heads_slow/', '/data/jfmadrig/VidTIMIT_lips_slow/', step=4)
    get_lips('/data/jfmadrig/hpedatasets/ict3DHP_heads/', '/data/jfmadrig/hpedatasets/ict3DHP_lips/')
    print("All Done!")
    exit()