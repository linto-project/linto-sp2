import os
from pathlib import Path
import cv2
import chainer
import numpy as np
from tools.detector_hyperface import HyperFaceModel, apply_hyperface
import tools.detector_hyperface as detector_hyperface


def create_lists(img_path, train_file, test_file, train_output='lists/train_list.txt', test_output='lists/test_list.txt'):
    f1 = open(train_file,'r')
    f2 = open(test_file,'r')

    train_list = f1.readlines()
    test_list = f2.readlines()

    f3 = open(train_output, 'w')
    f4 = open(test_output, 'w')

    clip_length = 16

    for line in train_list:
        name = line.split(' ')[0]
        image_path = img_path+name
        label = line.split(' ')[-1]
        images = os.listdir(image_path)
        nb = len(images) // clip_length
        for i in range(nb):
            f3.write(name+' '+ str(i*clip_length+1)+' '+label)


    for line in test_list:
        name = line.split(' ')[0]
        image_path = img_path+name
        label = line.split(' ')[-1]
        images = os.listdir(image_path)
        nb = len(images) // clip_length
        for i in range(nb):
            f4.write(name+' '+ str(i*clip_length+1)+' '+label)

    f1.close()
    f2.close()
    f3.close()
    f4.close()


def create_ucf101_list():

    img_path = '/home/tianz/datsets/ucfimgs/'
    f1 = 'ucfTrainTestlist/train_file.txt'
    f2 = 'ucfTrainTestlist/test_file.txt'
    create_lists(img_path, f1, f2)


def create_lr2_vid_list():

    img_path = ''
    f1 = 'lrs2TrainTestlist/train_file.txt'
    f2 = 'lrs2TrainTestlist/test_file.txt'
    create_lists(img_path, f1, f2)


def create_afew():

    img_path = ''
    f1 = 'afew-va/train_file.txt'
    f2 = 'afew-va/test_file.txt'
    create_lists(img_path, f1, f2)


def make_label(dirName, label = '#', train_output = 'lists/train_list.txt', test_output = 'lists/test_list.txt', maximage = 9999):

    # dirName = '/home/varun/Downloads'

    print("****************")

    clip_length = 16

    # Get the list of all files in directory tree at given path
    trainFiles = list()
    testFiles = list()
    count = 1
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        if len(filenames) == 0:# or '_45D' in dirpath or '_00D' in dirpath:
            continue
        # listOfFiles += [dirpath]
        # listOfFiles += [os.path.join(dirpath, file) for file in filenames]

        # filenames.sort(key=str.lower)

        nb = len(filenames) // clip_length
        nb = min(nb,maximage)
        for i in range(nb):
            if count % 4:
                trainFiles += [dirpath + ' ' + str(i * clip_length + 1) + ' ' + label]
            else:
                testFiles += [dirpath + ' ' + str(i * clip_length + 1) + ' ' + label]

        count += 1

    dirpath = Path(train_output).parent
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    with open(train_output, 'w') as f:
        for item in trainFiles:
            f.write("%s\n" % item)

    with open(test_output, 'w') as f:
        for item in testFiles:
            f.write("%s\n" % item)


def make_label_ibugs(dirName, label = '#', train_output = 'lists/train_list.txt', test_output = 'lists/test_list.txt', maximage=9999, max_seq=9999):

    # dirName = '/home/varun/Downloads'

    print("****************")

    clip_length = 16

    # Get the list of all files in directory tree at given path
    trainFiles = list()
    testFiles = list()
    count = 1
    n_seq = 0
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        if len(filenames) == 0: # or '_45D' in dirpath or '_00D' in dirpath:
            continue
        if not '-' in label:
            if '_00D' in dirpath:
                l = label
            elif '_45D' in dirpath:
                l = str(int(label) + 1)
            elif '_90D' in dirpath:
                l = str(int(label) + 2)
            elif '_270D' in dirpath:
                l = str(int(label) + 3)
            elif '_315D' in dirpath:
                l = str(int(label) + 4)
        else:
            l = str(abs(int(label)))
        # listOfFiles += [dirpath]
        # listOfFiles += [os.path.join(dirpath, file) for file in filenames]

        # filenames.sort(key=str.lower)
        filenames = [k for k in filenames if '.wav' not in k and 'flow' not in k]
        n_seq += 1

        nb = len(filenames) // clip_length
        nb = min(nb,maximage)
        for i in range(nb):
            if count % 4:
                trainFiles += [dirpath + ' ' + str(i * clip_length + 1) + ' ' + l]
            else:
                testFiles += [dirpath + ' ' + str(i * clip_length + 1) + ' ' + l]

            count += 1
            # if count > max_seq:
            #     break

        # if count > max_seq:
        #     break

    dirpath = Path(train_output).parent
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    with open(train_output, 'w') as f:
        for item in trainFiles:
            f.write("%s\n" % item)

    with open(test_output, 'w') as f:
        for item in testFiles:
            f.write("%s\n" % item)

    return  n_seq, count


def make_label_amicorpus(dirName, label = '#', train_output = 'lists/train_list.txt', test_output = 'lists/test_list.txt', maximage=9999, max_seq=9999):
    print("****************")

    # Load hyperface
    hyperface_module = HyperFaceModel()
    hyperface_module.train = False
    hyperface_module.report = False
    hyperface_module.backward = False
    chainer.serializers.load_npz("/local/users/jfmadrig/ownCloud/LinTo/src/tools/weights/model_epoch_190", hyperface_module)
    # Setup GPU
    config_gpu = 0
    if config_gpu >= 0:
        chainer.cuda.check_cuda_available()
        chainer.cuda.get_device(config_gpu).use()
        hyperface_module.to_gpu()
        xp = chainer.cuda.cupy
    else:
        xp = np

    clip_length = 16

    # Get the list of all files in directory tree at given path
    trainFiles = list()
    testFiles = list()
    count = 1
    n_seq = 0
    IMG_SIZE = (227, 227)

    for (dirpath, dirnames, filenames) in os.walk(dirName):
        if len(filenames) == 0: # or '_45D' in dirpath or '_00D' in dirpath:
            continue
        if not '-' in label:
            if '_00D' in dirpath:
                l = label
            elif '_45D' in dirpath:
                l = str(int(label) + 1)
            elif '_90D' in dirpath:
                l = str(int(label) + 2)
            elif '_270D' in dirpath:
                l = str(int(label) + 3)
            elif '_315D' in dirpath:
                l = str(int(label) + 4)
        else:
            l = str(abs(int(label)))
        # listOfFiles += [dirpath]
        # listOfFiles += [os.path.join(dirpath, file) for file in filenames]

        # filenames.sort(key=str.lower)
        filenames = [k for k in filenames if '.wav' not in k and 'flow' not in k]
        n_seq += 1

        filenames = filenames.sort()

        nb = len(filenames) // clip_length
        nb = min(nb,maximage)
        for i in range(nb):

            imgs = []
            for j in range(clip_length):
                frame = cv2.imread(dirpath + '/' + filenames[clip_length*i + j])
                img = frame.astype(np.float32) / 255.0  # [0:1]
                img = cv2.resize(img, IMG_SIZE)
                img = cv2.normalize(img, None, -0.5, 0.5, cv2.NORM_MINMAX)
                img = np.transpose(img, (2, 0, 1))
                imgs.append(img)

            for j in range(clip_length):
                frame = cv2.imread(dirpath + '/' + filenames[clip_length*i + j])
                poses = apply_hyperface(frame, None, None, hyperface_module, bh, xp)

            if 'CV5' in dirpath:
                testFiles += [dirpath + ' ' + str(i * clip_length + 1) + ' ' + l]
            else:
                trainFiles += [dirpath + ' ' + str(i * clip_length + 1) + ' ' + l]
            # if count > max_seq:
            #     break

        # if count > max_seq:
        #     break

    dirpath = Path(train_output).parent
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    with open(train_output, 'w') as f:
        for item in trainFiles:
            f.write("%s\n" % item)

    with open(test_output, 'w') as f:
        for item in testFiles:
            f.write("%s\n" % item)

    return n_seq, count


def make_label_ict3d(dir_path, gt_label_path='#', train_output='lists/train_list.txt', test_output='lists/test_list.txt',
                     maximage=9999, max_seq=9999):
    from numpy import genfromtxt
    from tools.utils import rotation_matrix, project_plane_yz, get_xpose
    import numpy as np
    import cv2

    def _draw_line(img, pt1, pt2, color, thickness=2):
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        cv2.line(img, pt1, pt2, color, int(thickness))

    print("****************")
    clip_length = 16

    # Get the list of all files in directory tree at given path
    trainFiles = list()
    testFiles = list()
    count = 1
    n_seq = 0


    for (dirpath, dirnames, filenames) in os.walk(dir_path):
        if len(filenames) == 0:  # or '_45D' in dirpath or '_00D' in dirpath:
            continue
        seq = dirpath.split('/')[-1]
        gt_path = gt_label_path + seq + '/' + "polhemusNorm.csv"
        gt_data = genfromtxt(gt_path, delimiter=',')

        init_pose = np.loadtxt(gt_label_path + seq + '/' + "initpose.txt", delimiter=',')

        n_seq += 1

        nb = len(filenames) // clip_length
        nb = min(nb, maximage)
        for i in range(nb):

            img_path = dirpath + '/{:06d}'.format(i * clip_length) + '.jpg'
            if not os.path.isfile(img_path):
                continue
            # img = cv2.imread(img_path)

            pose = gt_data[i*clip_length][3:]
            rotpose = rotation_matrix(-pose[2], pose[0], -pose[1])
            # rotinit = rotation_matrix(-init_pose[2], init_pose[0], -init_pose[1])
            # rotmat = np.transpose(rotpose).dot(rotinit)

            rotmat = rotpose
            pred = get_xpose(rotmat)
            l = np.argmax(pred)
            # print(pred)

            zvec = np.array([0, 0, 1], np.float32)
            yvec = np.array([0, 1, 0], np.float32)
            xvec = np.array([1, 0, 0], np.float32)
            zvec = project_plane_yz(rotmat.dot(zvec))
            yvec = project_plane_yz(rotmat.dot(yvec))
            xvec = project_plane_yz(rotmat.dot(xvec))

            # # Lower left
            # size = 30
            # idx = 0
            # org_pt = ((size + 5) * (2 * idx + 1), img.shape[0] - size - 5)
            # _draw_line(img, org_pt, org_pt + zvec * size, (255, 0, 0), 3)
            # _draw_line(img, org_pt, org_pt + yvec * size, (0, 255, 0), 3)
            # _draw_line(img, org_pt, org_pt + xvec * size, (0, 0, 255), 3)
            # cv2.imshow('result', img)
            # cv2.waitKey(1)
            if '02' in dirpath and test_output is not '':
                testFiles += [dirpath + ' ' + str(i * clip_length + 1) + ' ' + str(l)]
            else:
                trainFiles += [dirpath + ' ' + str(i * clip_length + 1) + ' ' + str(l)]
            if count > max_seq:
                break

        # if count > max_seq:
        #     break

    dirpath = Path(train_output).parent
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    with open(train_output, 'w') as f:
        for item in trainFiles:
            f.write("%s\n" % item)

    if test_output is not '':
        with open(test_output, 'w') as f:
            for item in testFiles:
                f.write("%s\n" % item)

    return n_seq, count


if __name__ == '__main__':
    # create_ucf101_list()
    # create_lr2_vid_list()
    # create_afew()
    # make_label('/data/jfmadrig/mvlrs_v1/pretrain_heads', '0',
    #            train_output = 'lrs2TrainTestlist/train_heads_list.txt',
    #            test_output = 'lrs2TrainTestlist/test_heads_list.txt')
    # make_label('/data/jfmadrig/VidTIMIT_heads', '1',
    #            train_output='lists/vid/train_heads_list.txt',
    #            test_output='lists/vid/test_heads_list.txt')
    # make_label('/data/jfmadrig/VidTIMIT_heads_slow', '1',
    #            train_output='lists/vid/train_heads_slow_short_list.txt',
    #            test_output='lists/vid/test_heads_slow_short_list.txt',
    #            maximage=100)
    # make_label('/data/jfmadrig/VidTIMIT_lips', '1',
    #            train_output='lists/vid/train_lips_list.txt',
    #            test_output='lists/vid/test_lips_list.txt')
    # make_label('/data/jfmadrig/mvlrs_v1/pretrain_lips', '0',
    #            train_output = 'lists/lrs2TrainTestlist/train_lips_list.txt',
    #            test_output = 'lists/lrs2TrainTestlist/test_lips_list.txt')
    # make_label('/data/jfmadrig/VidTIMIT_lips_slow', '1',
    #            train_output='lists/vid/train_lips_list_slow.txt',
    #            test_output='lists/vid/test_lips_list_slow.txt',
    #            maximage=200)
    #
    # make_label('/data/jfmadrig/hpedatasets/ict3DHP_lips', '1',
    #            train_output='lists/ict3DHP/train_lips_list.txt',
    #            test_output='lists/ict3DHP/test_lips_list.txt')

    #
    # make_label_ibugs('/data/jfmadrig/ibug-avs/Silence_Digits_lips', '0',
    #            train_output='lists/ibug-avs/Silence/train_lips_list_angle.txt',
    #            test_output='lists/ibug-avs/Silence/test_lips_list_angle.txt')
    #
    # make_label_ibugs('/data/jfmadrig/ibug-avs/Voice_Digits_lips', '0',
    #            train_output='lists/ibug-avs/Voice/train_lips_list_angle.txt',
    #            test_output='lists/ibug-avs/Voice/test_lips_list_angle.txt')
    # silence_train_output = 'lists/ibug-avs/Silence/train_lips_list_angle_c6.txt'
    # silence_test_output = 'lists/ibug-avs/Silence/test_lips_list_angle_c6.txt'
    # speaker_train_output='lists/ibug-avs/Voice/train_lips_list_angle_c6.txt'
    # speaker_test_output='lists/ibug-avs/Voice/test_lips_list_angle_c6.txt'

    # silence_train_output = 'lists/ibug-avs/Silence/train_heads_list_angle_c5.txt'
    # silence_test_output = 'lists/ibug-avs/Silence/test_heads_list_angle_c5.txt'
    # speaker_train_output='lists/ibug-avs/Voice/train_heads_list_angle_c5.txt'
    # speaker_test_output='lists/ibug-avs/Voice/test_heads_list_angle_c5.txt'
    #
    # make_label_ibugs('/data/jfmadrig/ibug-avs/Silence_Digits_heads', '5',
    #            train_output=silence_train_output,
    #            test_output=silence_test_output)
    #
    # make_label_ibugs('/data/jfmadrig/ibug-avs/Voice_Digits_heads', '0',
    #            train_output=speaker_train_output,
    #            test_output=speaker_test_output)

    dataset = 'ibugs'
    dataset = 'amicorpus'
    # dataset = 'ict3d'

    if dataset == 'ibugs':
        silence_train_output = 'lists/ibug-avs/Silence_v2/train_heads_list_angle_c5.txt'
        silence_test_output = 'lists/ibug-avs/Silence_v2/test_heads_list_angle_c5.txt'
        silence_dir_path = '/data/jfmadrig/ibug-avs/Silence_Digits_v2_head/'
        # silence_init_label = '-1' # Binary
        silence_init_label = '5'
        speaker_train_output='lists/ibug-avs/Voice_v2/train_heads_list_angle_c5.txt'
        speaker_test_output='lists/ibug-avs/Voice_v2/test_heads_list_angle_c5.txt'
        speaker_dir_path = '/data/jfmadrig/ibug-avs/Voice_Digits_v2_head/'
        # speaker_init_label = '-0' # binary
        speaker_init_label = '0'

        train_outpu_file = 'lists/ibug-avs/train_heads_list_angle_c10_v2.txt'
        test_outpu_file = 'lists/ibug-avs/test_heads_list_angle_c10_v2.txt'

    elif dataset == 'amicorpus':
        # silence_train_output = 'lists/amicorpus/Silence/train_heads_list_angle_binary.txt'
        # silence_test_output  = 'lists/amicorpus/Silence/test_heads_list_angle_binary.txt'
        # silence_dir_path   = '/data/jfmadrig/amicorpus/5fold_valid_Silence/CV1/IS1002b/video/Closeup1/'
        # speaker_train_output = 'lists/amicorpus/Speaking/train_heads_list_angle_binary.txt'
        # speaker_test_output  = 'lists/amicorpus/Speaking/test_heads_list_angle_binary.txt'
        # speaker_dir_path   = '/data/jfmadrig/amicorpus/5fold_valid_Speaking/CV1/IS1002b/video/Closeup1/'
        # train_outpu_file = 'lists/amicorpus/train_heads_list_binary.txt'
        # test_outpu_file  = 'lists/amicorpus/test_heads_list_binary.txt'
        silence_dir_path   = '/data/jfmadrig/amicorpus/5fold_valid_wholeshort_v2_Silence/'
        speaker_dir_path   = '/data/jfmadrig/amicorpus/5fold_valid_wholeshort_v2_Speaking/'

        silence_train_output = 'lists/amicorpus/Silence/train_orientation_head_list_cv5.txt'
        silence_test_output  = 'lists/amicorpus/Silence/test_orientation_head_list_cv5.txt'
        speaker_train_output = 'lists/amicorpus/Speaking/train_orientation_head_list_cv5.txt'
        speaker_test_output  = 'lists/amicorpus/Speaking/test_orientation_head_list_cv5.txt'
        train_outpu_file = 'lists/amicorpus/train_orientation_head_list_cv5.txt'
        test_outpu_file  = 'lists/amicorpus/test_orientation_head_list_cv5.txt'

        # silence_init_label = '5'
        # speaker_init_label = '0'
        silence_init_label = '-1' # Binary
        speaker_init_label = '-0' # binary
    elif dataset == 'ict3d':
        imgs_path   = '/data/jfmadrig/hpedatasets/ict3DHP_rgb_of_heads/'
        gt_path = '/data/jfmadrig/hpedatasets/ict3DHP_data/'
        train_output = 'lists/ict3DHP/orientation/train_heads_list.txt'
        test_output = ''
        make_label_ict3d(imgs_path, gt_path, train_output, test_output)
        exit()

    max_seq = 5000
    maximage= 100
    n_seq, max_seq = \
    make_label_amicorpus(speaker_dir_path, speaker_init_label,
                     train_output=speaker_train_output,
                     test_output=speaker_test_output,
                     maximage=maximage,
                     max_seq=max_seq)

    make_label_amicorpus(silence_dir_path, silence_init_label,
                    train_output=silence_train_output,
                    test_output=silence_test_output,
                     maximage=maximage,
                    max_seq=max_seq )
    ## Binary
    # make_label_ibugs('/data/jfmadrig/ibug-avs/Silence_Digits_v2_head/', '-1',
    #            train_output=silence_train_output,
    #            test_output=silence_test_output)
    #
    # make_label_ibugs('/data/jfmadrig/ibug-avs/Voice_Digits_v2_head/', '-0',
    #            train_output=speaker_train_output,
    #            test_output=speaker_test_output)

    filenames = [silence_train_output, speaker_train_output]
    with open(train_outpu_file, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    filenames = [silence_test_output, speaker_test_output]
    with open(test_outpu_file, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    # make_label_ibugs('/data/jfmadrig/ibug-avs/Static_Digits_heads', '0',
    #            train_output='lists/ibug-avs/Static/train_heads_list_angle.txt',
    #            test_output='lists/ibug-avs/Static/test_heads_list_angle.txt')
