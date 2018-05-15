from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

import itertools as it
import numpy as np
import os

# Naive approach saving labels as counting orders of data in folders
# (0) folder have errors and so on..
SIGNS = ["ERROR",
        "STOP",
        "TURN LEFT",
        "TURN RIGHT",
        "DO NOT TURN LEFT",
        "DO NOT TURN RIGHT",
        "ONE WAY",
        "SPEED LIMIT",
        "OTHER"]


'''
    
'''
def combine_rows(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    if PY3:
        output = it.zip_longest(fillvalue=fillvalue, *args)
    else:
        output = it.izip_longest(fillvalue=fillvalue, *args)
    return output

'''
'''
def grid_up(columns, images):
    images = iter(images)
    if PY3:
        image = next(images)
    else:
        image = images.next()
    fill_value = np.zeros_like(image)
    imgs = it.chain([image], images)
    rows = combine_rows(columns, imgs, fill_value)
    return np.vstack(map(np.hstack, rows))


'''
    clean previous output
'''
def clear_output():
    file_list = os.listdir('./outputDir/')
    print('file lists\n')
    print(file_list)
    for file_name in file_list:
        if '.png' in file_name:
            # remove previous output files
            os.remove(file_name)
