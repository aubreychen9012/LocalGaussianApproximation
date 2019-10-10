import numpy as np
import scipy.ndimage
import h5py
import glob
import random
import nibabel as nib
import pickle
from pdb import set_trace as bp

def rotate_3d_scipy(img, angles=None):
    dims = img.shape
    assert len(dims) >= 3
    random_angles = 0
    if not angles:
        angle_ax1 = random.uniform(-5, 5)
        angle_ax2 = random.uniform(-5, 5)
        angle_ax3 = random.uniform(-5, 5)
        random_angles=1
    else:
        angle_ax1, angle_ax2, angle_ax3 = angles
    img_rot = scipy.ndimage.interpolation.rotate(img, angle_ax1, mode='nearest',
                                                 axes=(0, 1), reshape=False)
    img_rot = scipy.ndimage.interpolation.rotate(img_rot, angle_ax2, mode='nearest',
                                                 axes=(0, 2), reshape=False)
    # rotate along x-axis
    img_rot = scipy.ndimage.interpolation.rotate(img_rot, angle_ax3, mode='nearest',
                                                 axes=(1, 2), reshape=False)
    if not random_angles:
        return img_rot
    else:
        return img_rot, [angle_ax1, angle_ax2, angle_ax3]

def create_dataset(augment=0):
    sub_name = \
        glob.glob('/scratch_net/bmicdl01/Data/CamCAN_unbiased/CamCAN/T1w/*_normalized_cropped*')
    #id = np.arange(len(sub_name))
    #np.random.shuffle(id)
    #sub_name = np.array(sub_name)[id]
    # test_sub = sub_name[:50]
    # with open("brats_test_list.p", 'wb') as f:
    #     pickle.dump([test_sub], f)

    sub = sub_name[50:250]
    sub_mask = ['mask'.join(i.split('normalized')) for i in sub]

    print("creating training data, {} subjects..".format(len(sub)))
    if not augment:
        data_file = h5py.File('/scratch_net/bmicdl01/Data/camcan_t1_train_set.hdf5','w')
    else:
        data_file = h5py.File('/scratch_net/bmicdl01/Data/camcan_t1_aug_train_set.hdf5', 'w')

    first = 1
    for i,j in zip(sub, sub_mask):
        print(i)
        nifti_img = nib.load(i)
        nifti_mask = nib.load(j)
        nifti_img = nifti_img.get_data()
        nifti_mask = nifti_mask.get_data()

        nifti_img = nifti_img.reshape(-1,200*200)
        nifti_mask = nifti_mask.reshape(-1, 200 * 200)

        assert len(nifti_img)==len(nifti_mask)

        if first:
            data_file.create_dataset('Scan', data=nifti_img,  maxshape=(None, 200*200))
            data_file.create_dataset('Mask', data=nifti_mask,  maxshape=(None, 200*200))
            first = 0
        else:
            data_file["Scan"].resize((data_file["Scan"].shape[0] + len(nifti_img)), axis=0)
            data_file["Scan"][-len(nifti_img):] = nifti_img
            data_file["Mask"].resize((data_file["Mask"].shape[0] + len(nifti_mask)), axis=0)
            data_file["Mask"][-len(nifti_mask):] = nifti_mask
        if augment:
            print("augment")
            nifti_img = nifti_img.reshape(-1, 200, 200)
            nifti_mask = nifti_mask.reshape(-1, 200, 200)
            len_img = len(nifti_img)
            for _ in range(3):
                nifti_img_rot, angles = rotate_3d_scipy(nifti_img)
                nifti_mask_rot = rotate_3d_scipy(nifti_mask, angles)
                nifti_img_rot = nifti_img_rot.reshape(-1, 200 * 200)
                nifti_mask_rot = nifti_mask_rot.reshape(-1, 200 * 200)
                data_file["Scan"].resize((data_file["Scan"].shape[0] + len_img), axis=0)
                data_file["Scan"][-len_img:] = nifti_img_rot
                data_file["Mask"].resize((data_file["Mask"].shape[0] + len_img), axis=0)
                data_file["Mask"][-len_img:] = nifti_mask_rot

    data_file.close()

    sub = sub_name[250:285]
    sub_mask = ['mask'.join(i.split('normalized')) for i in sub]
    #sub_seg = [i.replace("normalized_cropped_mask", "seg_cropped") for i in sub]
    print("creating validation data, {} subjects..".format(len(sub)))
    if not augment:
        data_file = h5py.File('/scratch_net/bmicdl01/Data/camcan_t1_val_set.hdf5', 'w')
    else:
        data_file = h5py.File('/scratch_net/bmicdl01/Data/camcan_t1_aug_val_set.hdf5', 'w')

    first=1
    for i, j in zip(sub, sub_mask):
        print(i)
        nifti_img = nib.load(i)
        nifti_mask = nib.load(j)
        nifti_img = nifti_img.get_data()
        nifti_mask = nifti_mask.get_data()
        #nifti_seg = nib.load(k).get_data()

        #idx = [i for i in range(len(nifti_seg)) if np.sum(i) == 0]

        nifti_img = nifti_img.reshape(-1, 200 * 200)
        nifti_mask = nifti_mask.reshape(-1, 200 * 200)

        assert len(nifti_img) == len(nifti_mask)

        if first:
            data_file.create_dataset('Scan', data=nifti_img, maxshape=(None, 200 * 200))
            data_file.create_dataset('Mask', data=nifti_mask, maxshape=(None, 200 * 200))
            first = 0
        else:
            data_file["Scan"].resize((data_file["Scan"].shape[0] + len(nifti_img)), axis=0)
            data_file["Scan"][-len(nifti_img):] = nifti_img
            data_file["Mask"].resize((data_file["Mask"].shape[0] + len(nifti_mask)), axis=0)
            data_file["Mask"][-len(nifti_mask):] = nifti_mask
        if augment:
            print("augment")
            len_img = len(nifti_img)
            for _ in range(3):
                nifti_img = nifti_img.reshape(-1, 200, 200)
                nifti_mask = nifti_mask.reshape(-1, 200, 200)

                nifti_img_rot, angles = rotate_3d_scipy(nifti_img)
                nifti_mask_rot = rotate_3d_scipy(nifti_mask, angles)
                nifti_img_rot = nifti_img_rot.reshape(-1, 200 * 200)
                nifti_mask_rot = nifti_mask_rot.reshape(-1, 200 * 200)
                data_file["Scan"].resize((data_file["Scan"].shape[0] + len_img), axis=0)
                data_file["Scan"][-len_img:] = nifti_img_rot
                data_file["Mask"].resize((data_file["Mask"].shape[0] + len_img), axis=0)
                data_file["Mask"][-len_img:] = nifti_mask_rot
    data_file.close()


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--augment', type=int, default=0)

    opt = parser.parse_args()
    create_dataset(opt.augment)


