import numpy as np


def dsc(pred, target):
    if np.sum(pred) + np.sum(target) == 0:
        return 1.
    return 2. * np.sum(pred * target) / (np.sum(pred) + np.sum(target))


def dsc_compute(labels, predictions, threshold, max_only=True):
    threshs = threshold
    cur_dsc = [dsc((predictions > thresh).astype(np.int), labels) for thresh in threshs]
    cur_dsc = np.array(cur_dsc)
    if max_only:
        # return max dice score, and threshold at the max dice
        return cur_dsc.max(), cur_dsc[np.argmax(cur_dsc)]
    # return array of dice scores within the threshold, max score and threshold at the max dice score
    return cur_dsc, cur_dsc.max(), cur_dsc[np.argmax(cur_dsc)]

