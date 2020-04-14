#!/bin/bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/cuda-9.0/lib64/

#tr="lists/amicorpus/train_heads_list_wholeshortV2.txt"
#ts="lists/amicorpus/test_heads_list_wholeshortV2.txt"
#sp="weights/ami_ws_v2/"
tr="lists/amicorpus/train_heads_list_cv1.txt"
ts="lists/amicorpus/test_heads_list_cv1.txt"
sp="weights/bs14/ami_ws_cv1/"
#tr="lists/amicorpus/train_heads_list_cv125_close12.txt"
#ts="lists/amicorpus/test_heads_list_cv125_close12.txt"
#sp="weights/ami_ts_cv125_close12/"
#tr="None"
#ts="None"
#sp="None"
eval="yes"
bs="14"
ep="21"


python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet2D_concat -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None

#bs="20"
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
#bs="15"
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
##
##ep="20"
##python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
##python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
##python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s








tr="lists/amicorpus/train_heads_list_cv2.txt"
ts="lists/amicorpus/test_heads_list_cv2.txt"
sp="weights/bs14/ami_ws_cv2/"
eval="yes"
bs="14"
ep="21"


python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet2D_concat -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s





tr="lists/amicorpus/train_heads_list_cv3.txt"
ts="lists/amicorpus/test_heads_list_cv3.txt"
sp="weights/bs14/ami_ws_cv3/"
eval="yes"
bs="14"
ep="21"


python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet2D_concat -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s




tr="lists/amicorpus/train_heads_list_cv4.txt"
ts="lists/amicorpus/test_heads_list_cv4.txt"
sp="weights/bs14/ami_ws_cv4/"
eval="yes"
bs="14"
ep="21"


python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet2D_concat -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s





tr="lists/amicorpus/train_heads_list_cv5.txt"
ts="lists/amicorpus/test_heads_list_cv5.txt"
sp="weights/bs14/ami_ws_cv5/"
eval="yes"
bs="14"
ep="21"


python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet2D_concat -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s














echo 'End'