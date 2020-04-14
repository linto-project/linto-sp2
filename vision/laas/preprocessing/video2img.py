import cv2
import os

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
                    cv2.imwrite(save_name+str(10000+fps_count)+'.jpg',frame)
                    fps_count += 1


def mvlrs_v1_dataset():
    video_path = '/data/jfmadrig/mvlrs_v1/pretrain/'
    save_path = '/data/jfmadrig/mvlrs_v1/pretrain_imgs/'

    action_list = sorted(os.listdir(video_path))
    limit = 1000
    for indx , action in enumerate(action_list):
        if not os.path.exists(save_path + action):
            os.mkdir(save_path + action)
        video_list = os.listdir(video_path + action)
        for video in video_list:
            prefix, suffix = video.split('.')
            if suffix  == 'txt':
                continue
            if not os.path.exists(save_path + action + '/' + prefix):
                os.mkdir(save_path + action + '/' + prefix)
            save_name = save_path + action + '/' + prefix + '/'
            video_name = video_path + action + '/' + video
            cap = cv2.VideoCapture(video_name)
            fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_count = 0
            for i in range(fps):
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(save_name + str(10000 + fps_count) + '.jpg', frame)
                    fps_count += 1

        print(video_path + action + " Done")

        if indx == limit :
            break


if __name__ == '__main__':
    # ucf_dataset()
    mvlrs_v1_dataset()