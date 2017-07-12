# NumPy for numerical computing
import numpy as np
np.random.seed(123)
import random
random.seed(123)

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_column', 100)

import os
import gc
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

from tags import Tags

PLANET_KAGGLE_ROOT = '/data/planet-data/'
if not os.path.exists(PLANET_KAGGLE_ROOT):
    PLANET_KAGGLE_ROOT = '/Users/jiayou/Documents/Kaggle Data/Amazon'

N_TAGS = 17
N_TRAIN = 40479
N_TEST_T = 40669
N_TEST_F = 20522
N_TEST = N_TEST_T + N_TEST_F

def load_train_image(n, tif=False, dbg=False):
    if tif:
        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, 'train-tif-v2', 'train_{}.tif'.format(n)))
    else:
        path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg', 'train_{}.jpg'.format(n)))
    if os.path.exists(path):
        img = io.imread(path)
#         if dbg:
#             plt.figure()
#             plt.imshow(img)
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

def get_training_data(file_ids, tif=False, dbg=False, verbose=False):
    if verbose:
        print('Getting {} training images...'.format(len(file_ids)))
    X_train = np.zeros((len(file_ids), 256, 256, 4 if tif else 3)).astype('float16')
    for i in range(len(file_ids)):
        X_train[i,:,:,:] = preprocess_image(load_train_image(file_ids[i], tif=tif, dbg=dbg))
        if verbose and i % 100 == 0:
            print('Got {} images'.format(i+1))
    if verbose:
        print('Done')

    y_train = Tags().y_train(file_ids)
    if dbg:
        print(y_train)

    return (X_train, y_train)

def get_test_data(file_ids):
    X_train = np.array([preprocess_image(load_test_image(fname)) for fname in file_ids])
    return X_train

def augment(im, orient = None):
    if orient is None:
        mirror = random.randint(0, 1)
        rotate = random.randint(0, 3)
    else:
        mirror = orient[0]
        rotate = orient[1]
    im = np.rot90(im, rotate, (0, 1))
    if mirror:
        im = np.flip(im, 1)
    return im

class Data:
    def __init__(self, tif=False, toy=None, train=[0,1,2,3,4]):
        n = N_TRAIN
        if toy is not None:
            n = toy

        self.c = 4 if tif else 3

        print('Loading data...')
        self.X = [0] * 5
        self.y = [0] * 5
        for i in train:
            if tif:
                self.X[i] = np.load('X.{}.npy'.format(i))
                self.y[i] = np.load('y.{}.npy'.format(i))
            else:
                self.X[i], self.y[i] = get_training_data(
                    [x for x in range(n) if x % 5 == i], tif=tif, verbose=True)
            print('Loaded fold {}.'.format(i))

    def gen_train(self, batch_size, val=0):
        while 1:
            f = val
            while f == val:
                f = random.randint(0, 4)
            yield self.data_from_fold(f, batch_size)

    def gen_val(self, batch_size, val=0):
        f = val
        while 1:
            ids = np.random.randint(0, len(self.y[f]), size=batch_size).tolist()
            ids.sort()
            yield (self.X[f][ids,:,:,:], self.y[f][ids,:])

    def gen_test(self, batch_size):
        start = 0
        while start < N_TEST:
            end = min(start + batch_size, N_TEST)
            yield get_test_data(range(start, end))
            start = end

    def data_from_fold(self, f, batch_size):
        ids = np.random.randint(0, len(self.y[f]), size=batch_size).tolist()
        ids.sort()
        X = np.zeros((len(ids), 256, 256, self.c))
        for i in range(len(ids)):
            X[i,:,:,:] = augment(self.X[f][ids[i],:,:,:].reshape((256, 256, self.c))).reshape((1, 256, 256, self.c))
        return (X, self.y[f][ids,:])

    def get_fold(self, f=0):
        return (self.X[f], self.y[f])




def get_training_file_ids(draw_size):
    file_ids = np.random.randint(0, N_TRAIN, size=draw_size).tolist()
    for i in range(len(file_ids)):
        if file_ids[i] % 5 == 0:
            file_ids[i] = (file_ids[i] + 1) % N_TRAIN
    return file_ids

def get_calib_params():
    draw_size = 1000
    file_ids = get_training_file_ids(draw_size)
    ref_color = [[], [], [], []]

    for i in range(draw_size):
        current_im = io.imread(os.path.join(PLANET_KAGGLE_ROOT, 'train-tif-v2', 'train_{}.tif'.format(file_ids[i])))
        flatten_im = current_im.reshape((-1, 4))
        for j in range(4):
            ref_color[j] += flatten_im[:, j].tolist()

    ref_color = np.array(ref_color)
    ref_param = np.zeros((2, 4))
    ref_param[0,:] = ref_color.mean(axis = 1)
    ref_param[1,:] = ref_color.std(axis = 1)
    return ref_param