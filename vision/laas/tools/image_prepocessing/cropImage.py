# Panorama to cubemap (Python 3)
import numpy as np
import sys, math, os
import cv2
from tools.cubemap import cubemap

def cropping(img, method):
    if method == 0: 
        return cubemap(img)
    elif method == 1:
        height = img.shape[0]
        width = img.shape[1]
        SIZE = int(height / 2)
        HSIZE = int(SIZE / 2.0)
        return img[HSIZE:3*HSIZE , :]
    elif method == 2:
        height = img.shape[0]
        width = img.shape[1]
        SIZE = int(height / 2)
        HSIZE = int(SIZE / 2.0)
        return np.concatenate((img[HSIZE:3*HSIZE , :], img[HSIZE:3*HSIZE , 0:int(width/16)]), axis=1)
