__author__ = 'shrprasha'

import os
import io
import json
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


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
