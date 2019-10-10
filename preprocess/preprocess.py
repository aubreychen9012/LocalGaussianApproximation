import numpy as np
import skimage.measure
import scipy
from scipy.misc import imresize

def generate_patch(img_3d, patch_size, step_size, threshold):
    img = np.asarray(img_3d)
    min_ = img.min()
    l_patch = []
    _,x,y = img.shape
    for im in img:
        for xidx in np.arange(0, x-patch_size+step_size, step_size):
            for yidx in np.arange(0, y-patch_size+step_size, step_size):
               patch = im[xidx:xidx+patch_size, yidx:yidx+patch_size]
               if (patch==min_).sum()<patch_size**2*threshold:
                   l_patch.append(patch)
    return np.asarray(l_patch)

## AvgPool-like effect
def generate_low_resolution_images(img_3d):
    _,original_x, orignal_y = img_3d.shape
    img_3d_lowres = []
    for im in img_3d:
        im_lowres = skimage.measure.block_reduce(im, (2,2), np.mean)
        img_3d_lowres.append(im_lowres)
    return np.asarray(img_3d_lowres)

def resize(img_3d,scale, method):
    img=img_3d
    res = []
    for im in img:
        i_ = imresize(im,scale, interp=method)
        i_re = (i_/255.0)*(im.max()-im.min())+im.min()
        res.append(i_re)
    return np.asarray(res)

def resize_2d(img_2d,scale):
    img=img_2d
    #res = []
    #for im in img:
    i_ = imresize(img,scale, interp='nearest')
    i_re = (i_/255.0)*(img.max()-img.min())+img.min()
    #res.append(i_re)
    return i_re

def lowres_level(dt, level=3):
    res = []
    for img in dt:
        lv=0
        while lv<level:
            img = generate_low_resolution_images(img)
            lv+=1
        res.extend(img)
    return np.asarray(res)

