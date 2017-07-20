
# coding: utf-8

# # image preprocessing

# In[4]:

# print_function for compatibility with Python 3
from __future__ import print_function
print('print function is ready to serve')

# NumPy for numerical computing
import numpy as np
np.random.seed(123)

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_column', 100)

import os
import gc
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean


# In[25]:

PLANET_KAGGLE_ROOT = '/data/planet-data/'
if not os.path.exists(PLANET_KAGGLE_ROOT):
    PLANET_KAGGLE_ROOT = '/Users/jiayou/Documents/Kaggle Data/Amazon'

N_TAGS = 17
N_TRAIN = 40479
# N_TRAIN = 10
N_USE = 32000
# N_USE = 20
N_TEST_T = 40669
N_TEST_F = 20522
N_TEST = N_TEST_T + N_TEST_F

def load_image(filename):
    '''Look through the directory tree to find the image you specified
    (e.g. train_10.tif vs. train_10.jpg)'''
    for dirname in os.listdir(PLANET_KAGGLE_ROOT):
        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, dirname, filename))
        if os.path.exists(path):
            print('Found image {}'.format(path))
            return io.imread(path)
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))

def load_train_image(n, tif=False, dbg=False):
    if tif:
        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, 'train-tif-v2', 'train_{}.tif'.format(n)))
    else:
        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg', 'train_{}.jpg'.format(n)))
    if os.path.exists(path):
        img = io.imread(path)
        return img
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))

def load_test_image(n):
    path = None
    if n < N_TEST_T:
        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, 'test-jpg', 'test_{}.jpg'.format(n)))
    else:
        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, 'test-jpg-additional', 'file_{}.jpg'.format(n - N_TEST_T)))
    if os.path.exists(path):
        return io.imread(path)
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))


calib_params = np.array(
    [[ 4953.06200497,  4238.24180873,  3039.04404623,  6387.04264221],
     [ 1692.87422811,  1528.24629706,  1576.04566834,  1804.99976545]]
)

def preprocess_image(img):
    img = img.astype('float16')
#     img = downscale_local_mean(img, (4, 4, 1))

    if img.shape[2] == 3:
        # jpg
        img = img / 255 - 0.5
    else:
        # tif
        for i in range(4):
            img[:,:,i] = (img[:,:,i] - calib_params[0,i]) / 1500

    return img


# In[26]:

train_labels = pd.read_csv(os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv'))

label_list = []
for tag_str in train_labels.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)

label_map = {}
for i in range(len(label_list)):
    label_map[label_list[i]] = i
label_idx = {v: k for k, v in label_map.items()}


# In[27]:

def get_training_data(file_ids, tif=False, dbg=False, verbose=False):
    if verbose:
        print('Getting {} training images...'.format(len(file_ids)))
    X_train = np.zeros((len(file_ids), 256, 256, 4 if tif else 3)).astype('float16')
    X_train = X_train.astype('float16')
    for i in range(len(file_ids)):
        X_train[i,:,:,:] = preprocess_image(load_train_image(file_ids[i], tif=tif, dbg=dbg))
        if verbose and i % 100 == 0:
            print('Got {} images'.format(i+1))
    if verbose:
        print('Done')

    y_train = np.array([[0. for i in range(N_TAGS)] for j in file_ids])
    for i in range(len(file_ids)):
        tags = train_labels.tags[file_ids[i]]
        if dbg:
            print(file_ids[i], tags)
        for tag in tags.split(' '):
            if not tag in label_map:
                continue
            y_train[i][label_map[tag]] = 1.
    if dbg:
        print(y_train)

    return (X_train, y_train)


# In[ ]:

for i in range(5):
    file_ids = [x for x in range(N_TRAIN) if x % 5 == i]
    X_train_i, y_train_i = get_training_data(file_ids, tif = True, verbose = True)
    np.save('X.{}.npy'.format(i), X_train_i)
    np.save('y.{}.npy'.format(i), y_train_i)
    gc.collect()


# In[28]:

# x,y = get_training_data([0], tif=True)


# In[29]:

# x.dtype


# In[ ]:



