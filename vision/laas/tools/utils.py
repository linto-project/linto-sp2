from random import randint
import numpy as np
import math


def project_plane_yz(vec):
    x = vec.dot(np.array([0, 1, 0], dtype=np.float32))
    y = vec.dot(np.array([0, 0, 1], dtype=np.float32))
    return np.array([x, -y], dtype=np.float32)  # y flip


def rotation_matrix(rad_x, rad_y, rad_z):
    cosx, cosy, cosz = math.cos(rad_x), math.cos(rad_y), math.cos(rad_z)
    sinx, siny, sinz = math.sin(rad_x), math.sin(rad_y), math.sin(rad_z)
    rotz = np.array([[cosz, -sinz, 0],
                     [sinz, cosz, 0],
                     [0, 0, 1]], dtype=np.float32)
    roty = np.array([[cosy, 0, siny],
                     [0, 1, 0],
                     [-siny, 0, cosy]], dtype=np.float32)
    rotx = np.array([[1, 0, 0],
                     [0, cosx, -sinx],
                     [0, sinx, cosx]], dtype=np.float32)
    return rotx.dot(roty).dot(rotz)


def get_xpose( pose ):
    # parallel projection (something wrong?)
    if pose.ndim == 1:
        rotmat = rotation_matrix(-pose[0], -pose[1], -pose[2])
    else:
        rotmat = pose
    zvec = np.array([0, 0, 1], np.float32)
    yvec = np.array([0, 1, 0], np.float32)
    xvec = np.array([1, 0, 0], np.float32)
    # zvec = drawing._project_plane_yz(rotmat.dot(zvec))
    # yvec = drawing._project_plane_yz(rotmat.dot(yvec))
    xvec = project_plane_yz(rotmat.dot(xvec))

    kappa = 32
    # Original
    # mu = [-1, -sqrt(2)/2, 0, sqrt(2)/2, 1]

    # Following convention (angle->label) 0->0, 45->1 90->2  270->3  315->4
    mu = [0, math.sqrt(2) / 2, 1, -1, -math.sqrt(2) / 2, ]

    x = xvec[0]
    from scipy.special import i0
    val = np.exp(kappa * np.cos(mu - x)) / (2 * np.pi * i0(kappa))
    nval = np.linalg.norm(val, ord=2)
    nval = np.sum(val)
    val = val / nval
    return val

def get_num_samples(test_file, train_file):
    f1 = open(train_file, 'r')
    f2 = open(test_file, 'r')
    lines = f1.readlines()
    f1.close()
    train_samples = len(lines)
    lines = f2.readlines()
    f2.close()
    val_samples = len(lines)
    return train_samples, val_samples


# Union of two rectangles
# Rect format: x, y, w, y
def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

# Intersection of two rectangles
# Rect format: x, y, w, y
def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return (0,0,0,0)
    return (x, y, w, h)


# Box format: startY, startX, endY, endX
# Rect format: x, y, w, y
def boxp1p2_to_rect(box):
    return (box[1], box[0], box[3] - box[1], box[2] - box[0])

# Intersection over Union of two rectangles
# Rect format: x, y, w, y
def IoU(det_box, box):
    x, y, w, h = intersection(det_box, box)
    if w == 0:
        return 0

    area_int = w * h
    x, y, w, h = union(det_box, box)
    area_uni = w * h

    return area_int / area_uni

def create_colors(n):
    ret = []
    for i in range(n):
        ret.append((randint(0, 255),randint(0, 255),randint(0, 255)))
    return ret


def noisy(noise_typ,image):
    if noise_typ is "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ is "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ is "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ is "speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy
