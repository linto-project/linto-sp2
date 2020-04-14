#!/bin/bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/cuda-9.0/lib64/

#tr="lists/amicorpus/train_heads_list_wholeshortV2.txt"
#ts="lists/amicorpus/test_heads_list_wholeshortV2.txt"
#sp="weights/ami_ws_v2/"
#tr="lists/amicorpus/train_heads_list_cv125_close12.txt"
#ts="lists/amicorpus/test_heads_list_cv125_close12.txt"
#sp="weights/ami/"
tr="lists/ibug-avs/train_heads_list_angle_c10.txt"
ts="lists/ibug-avs/test_heads_list_angle_c10.txt"
sp="weights/ibugs/c10/"
#tr="None"
#ts="None"
#sp="None"
eval="yes"
bs="10"
ep="10"

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet2D_concat -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None

python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None
python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=None


#python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=C3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
#python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
#python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=resnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
#
#python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
#python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
#python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
#
#ep="20"
#python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sC3D -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
#python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_18 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s
#python3 trainer_audio_video.py -tr=$tr -ts=$ts -sp=$sp -eval=$eval -net_video=sresnet3D_34 -eps=$ep -ds=0 -df=1 -bs=$bs -net_audio=resnet34s

echo 'End'