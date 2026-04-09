import numpy as np
from sklearn.metrics import roc_curve


def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1 = tpr + fpr - 1.0

    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    return err, best_th


def performances(val_scores, val_labels, val_threshold=None):
    assert len(val_scores) == len(val_labels)
    count = len(val_scores)
    num_real = 1e-5
    num_fake = 1e-5
    for label in val_labels:
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    if val_threshold is None:
        fpr, tpr, threshold = roc_curve(val_labels, val_scores, pos_label=1)
        val_err, val_threshold = get_err_threhold(fpr, tpr, threshold)

    type1 = len([s for s, l in zip(val_scores, val_labels) if s <= val_threshold and l == 1])
    type2 = len([s for s, l in zip(val_scores, val_labels) if s > val_threshold and l == 0])

    val_acc = 1 - (type1 + type2) / count
    val_apcer = type2 / num_fake
    val_bpcer = type1 / num_real
    val_acer = (val_apcer + val_bpcer) / 2.0

    return val_threshold, val_acc, val_apcer, val_bpcer, val_acer
