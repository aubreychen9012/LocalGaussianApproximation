import numpy as np
import h5py
import glob
import random
import nibabel as nib
import pickle
from pdb import set_trace as bp

def create_dataset():
    sub_name = glob.glob('/scratch_net/bmicdl01/Data/BraTS_unbiased_aligned/T2w-rigid-to-mni/*/*_normalized_cropped*')
    id = np.arange(len(sub_name))
    np.random.shuffle(id)
    sub_name = np.array(sub_name)[id]
    test_sub = sub_name[:50]
    with open("brats_test_list.p", 'wb') as f:
        pickle.dump([test_sub], f)

    sub = sub_name[50:250]
    sub_mask = ['mask'.join(i.split('normalized')) for i in sub]
    sub_seg = [i.replace("normalized_cropped_mask", "seg_cropped") for i in sub]

    print("creating training data, {} subjects..".format(len(sub)))
    data_file = h5py.File('/scratch_net/bmicdl01/Data/brats_all_train.hdf5','w')

    first = 1
    for i,j,k in zip(sub, sub_mask, sub_seg):
        print(i,j,k)
        nifti_img = nib.load(i)
        nifti_mask = nib.load(j)
        nifti_img = nifti_img.get_data()
        nifti_mask = nifti_mask.get_data()
        nifti_seg = nib.load(k).get_data()

        idx = [l for l in range(len(nifti_seg))]# if np.sum(nifti_seg[l])==0]

        nifti_img = nifti_img[idx].reshape(-1,200*200)
        nifti_mask = nifti_mask[idx].reshape(-1, 200 * 200)
        nifti_seg = nifti_seg[idx].reshape(-1,200*200)

        assert len(nifti_img)==len(nifti_mask)

        if first:
            data_file.create_dataset('Scan', data=nifti_img,  maxshape=(None, 200*200))
            data_file.create_dataset('Mask', data=nifti_mask,  maxshape=(None, 200*200))
            data_file.create_dataset("Seg", data=nifti_seg, maxshape=(None, 200*200))
            first = 0
        else:
            data_file["Scan"].resize((data_file["Scan"].shape[0] + len(nifti_img)), axis=0)
            data_file["Scan"][-len(nifti_img):] = nifti_img
            data_file["Mask"].resize((data_file["Mask"].shape[0] + len(nifti_mask)), axis=0)
            data_file["Mask"][-len(nifti_mask):] = nifti_mask
            data_file["Seg"].resize((data_file["Seg"].shape[0] + len(nifti_seg)), axis=0)
            data_file["Seg"][-len(nifti_seg):] = nifti_seg

    data_file.close()

    sub = sub_name[250:]
    sub_mask = ['mask'.join(i.split('normalized')) for i in sub]
    sub_seg = [i.replace("normalized_cropped_mask", "seg_cropped") for i in sub]
    print("creating validation data, {} subjects..".format(len(sub)))
    data_file = h5py.File('/scratch_net/bmicdl01/Data/brats_all_val.hdf5', 'w')

    first=1
    for i, j, k in zip(sub, sub_mask, sub_seg):
        print(i, j, k)
        nifti_img = nib.load(i)
        nifti_mask = nib.load(j)
        nifti_img = nifti_img.get_data()
        nifti_mask = nifti_mask.get_data()
        nifti_seg = nib.load(k).get_data()

        idx = [l for l in range(len(nifti_seg))] # if np.sum(nifti_seg[l]) == 0]

        nifti_img = nifti_img[idx].reshape(-1, 200 * 200)
        nifti_mask = nifti_mask[idx].reshape(-1, 200 * 200)
        nifti_seg = nifti_seg[idx].reshape(-1, 200 * 200)

        assert len(nifti_img) == len(nifti_mask)

        if first:
            data_file.create_dataset('Scan', data=nifti_img, maxshape=(None, 200 * 200))
            data_file.create_dataset('Mask', data=nifti_mask, maxshape=(None, 200 * 200))
            data_file.create_dataset('Seg', data=nifti_seg, maxshape=(None, 200 * 200))
            first = 0
        else:
            data_file["Scan"].resize((data_file["Scan"].shape[0] + len(nifti_img)), axis=0)
            data_file["Scan"][-len(nifti_img):] = nifti_img
            data_file["Mask"].resize((data_file["Mask"].shape[0] + len(nifti_mask)), axis=0)
            data_file["Mask"][-len(nifti_mask):] = nifti_mask
            data_file["Seg"].resize((data_file["Seg"].shape[0] + len(nifti_seg)), axis=0)
            data_file["Seg"][-len(nifti_mask):] = nifti_seg
    data_file.close()


if __name__=="__main__":
    create_dataset()


