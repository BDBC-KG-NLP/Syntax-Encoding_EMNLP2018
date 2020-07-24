import os
import io
import json
import numpy as np
from sklearn import metrics


def create_opdir(opdir):
    if not os.path.exists(opdir):
        os.makedirs(opdir)


def create_dumps_json(obj, fpath):
    with open(fpath, "w") as fhandle:
        json.dump(obj, fhandle)


def get_lines_in_file(fpath):
    """
    Uses a generator to read a large file lazily
    """
    with io.open(fpath, encoding='utf-8') as fhandle:
        while True:
            data = fhandle.readline()
            if not data:
                break
            yield data


def get_lines_in_file_small(fpath, remove_empty=True, encoding='utf-8'):
    with io.open(fpath, encoding=encoding) as fhandle:
        lines = fhandle.readlines()
    cleaned = []
    if remove_empty:
        for line in lines:
            if line.strip():
                cleaned.append(line)
        return cleaned
    return lines


def get_files_in_folder(folder, extension=None, fname_contains=None):
    all_files = sorted(next(os.walk(folder))[2])
    if ".DS_Store" in all_files:
        all_files.remove(".DS_Store")
    if extension:
        extension = "." + extension
        for fname in all_files:
            if extension not in fname:
                all_files.remove(fname)
    if fname_contains:
        for fname in all_files:
            if fname_contains not in fname:
                all_files.remove(fname)
    return all_files


def ctr_remove_below_threshold(ctr, threshold):
    for key, count in reversed(ctr.most_common()):
        if count < threshold:
            del ctr[key]
        else:
            break


def get_evaluation(y_true, y_prob, list_metrics):
    """
    y_true和y_prob都是CPU版本，为什么呢？
    """
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))

    if 'pre-rec-f1' in list_metrics:
        pre, rec, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred)
        output['pre'] = pre
        output['rec'] = rec
        output['f1'] = f1
    return output