# Panorama to cubemap (Python 3)
import numpy as np
from scipy import ndimage, misc
import sys, math, os
from PIL import Image
import cv2


def cubemap(im):
    # im = ndimage.imread(filename)
    height = im.shape[0]    
    width = im.shape[1]   
    SIZE = int(height / 2)
    #print(SIZE)
    HSIZE = SIZE / 2.0
    side_im = np.zeros((SIZE, SIZE), np.uint8)
    color_side = np.zeros((SIZE, SIZE, 3), np.uint8)
    pids = []
    vimg = []
    for i in range(0,6):
        # Multiple process to go faster!
        pid = os.fork()
        if pid != 0:
            #  Keep track of our children
            pids.append(pid)
            continue

        #  This is numpy's way of visiting each point in an ndarray, I guess its fast...
        it = np.nditer(side_im, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            # Axis
            axA = it.multi_index[0]
            axB = it.multi_index[1]
            # Color is an axis, so we visit each point 3 times for R,G,B actually...
       
            # Here for each face we decide what each axis represents, x, y or z. 
            z = -axA + HSIZE
            
            if i == 0:
                x = HSIZE
                y = -axB + HSIZE
            elif i == 1:
                x = -HSIZE
                y = axB - HSIZE
            elif i == 2:
                x = axB - HSIZE
                y = HSIZE
            elif i == 3:
                x = -axB + HSIZE
                y = -HSIZE
            elif i == 4:
                z = HSIZE
                x = axB - HSIZE
                y = axA - HSIZE
            elif i == 5:
                z = -HSIZE
                x = axB - HSIZE
                y = -axA + HSIZE
        
            # Now that we have x,y,z for point on plane, convert to spherical
            r = math.sqrt(float(x*x + y*y + z*z))
            theta = math.acos(float(z)/r)
            phi = -math.atan2(float(y),x)
            
            # Now that we have spherical, decide which pixel from the input image we want.
            ix = int((im.shape[1]-1)*phi/(2*math.pi))
            iy = int((im.shape[0]-1)*(theta)/math.pi)
            # This is faster than accessing the whole tuple! WHY???
            r = im[iy, ix, 0]
            g = im[iy, ix, 1]
            b = im[iy, ix, 2]
            color_side[axA, axB, 0] = r
            color_side[axA, axB, 1] = g
            color_side[axA, axB, 2] = b

            it.iternext()
        # Save output image using prefix, type and index info.
        # pimg = Image.fromarray(color_side)
        # vimg.append(cv2.cvtColor(color_side, cv2.COLOR_BGR2RGB))
        # pimg.save(os.path.join('./', "%s%d.%s"%('side_',i,'.jpg')), quality=85)
        
        # Children Exit here
        sys.exit(0)

    #  Thise seems to work better than waitpid(-1, 0), in that case sometimes the
    #  files still don't exist and we get an error.
    for pid in pids: 
        os.waitpid(pid, 0)
    return vimg

