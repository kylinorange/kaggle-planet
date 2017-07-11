
# coding: utf-8

# # Common

# In[360]:

# print_function for compatibility with Python 3
from __future__ import print_function
print('print function is ready to serve')

# NumPy for numerical computing
import numpy as np
np.random.seed(123)
import random
random.seed(123)

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_column', 100)

# Matplotlib for visualization
# from matplotlib import pyplot as plt

# # display plots in the notebook
# get_ipython().magic('matplotlib inline')

import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD, Adam

import os
import gc
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

import sys
sys.path.append('./utils')


# In[361]:

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


# In[362]:

# img = load_train_image(1)
# plt.imshow(img)
# img.shape


# In[363]:

train_labels = pd.read_csv(os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv'))
train_labels.head()


# In[364]:

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

weather_labels = [0, 3, 9, 10]
weather_labels = []

thres = [0.24, 0.22, 0.25, 0.23, 0.26, 0.19, 0.24, 0.23, 0.1, 0.16, 0.14,
         0.2, 0.17, 0.33, 0.04, 0.12, 0.07]

def pred_to_tags(y):
    maxw = 0
    w = 3
    tags = []
    for i in range(N_TAGS):
        tag = label_idx[i]
        if i in weather_labels:
            if y[i] > maxw:
                maxw = y[i]
                w = i
        else:
            if y[i] >= thres[i]:
                tags.append(tag)
#     tags.append(label_idx[w])
    return ' '.join(tags)

def output(pred):
    result = pd.DataFrame({
        'image_name':
            ['test_{}'.format(i) for i in range(N_TEST_T)] + ['file_{}'.format(i) for i in range(N_TEST_F)],
        'tags': ['' for i in range(N_TEST)]
    })
    for i in range(len(pred)):
        current_pred = pred[i]
        current_tag = pred_to_tags(current_pred)
        result.iat[i, 1] = current_tag
    return result

label_idx


# In[365]:

def get_training_data(file_ids, tif=False, dbg=False, verbose=False):
    if verbose:
        print('Getting {} training images...'.format(len(file_ids)))
    X_train = np.zeros((len(file_ids), 256, 256, 4 if tif else 3))
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

def get_test_data(file_ids):
    X_train = np.array([preprocess_image(load_test_image(fname)) for fname in file_ids])
    return X_train

def gen_training_data(batch_size, dbg=False):
    while 1:
        file_ids = np.random.randint(0, N_TRAIN, size=batch_size).tolist()
        for i in range(len(file_ids)):
            if file_ids[i] % 5 == 0:
                file_ids[i] = (file_ids[i] + 1) % N_TRAIN
        file_ids.sort()
        if dbg:
            print('file ids: ', file_ids)
        yield get_training_data(file_ids, dbg)


def gen_validation_data(batch_size):
    while 1:
        file_ids = (np.random.randint(0, int(N_TRAIN / 5), size=batch_size) * 5).tolist()
        file_ids.sort()
        yield get_training_data(file_ids)

def gen_test_data(batch_size):
    start = 0
    while start < N_TEST:
        end = min(start + batch_size, N_TEST)
        yield get_test_data(range(start, end))
        start = end


# In[366]:

class Data:
    def __init__(self, tif=False):
        self.c = 4 if tif else 3
        print('Loading data...')
        self.X = [0] * 5
        self.y = [0] * 5
        for i in range(5):
            if tif:
                self.X[i] = np.load('X.{}.npy'.format(i))
                self.y[i] = np.load('y.{}.npy'.format(i))
            else:
                self.X[i], self.y[i] = get_training_data(
                    [x for x in range(N_TRAIN) if x % 5 == i], tif=tif, verbose=True)
                gc.collect()
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

    def data_from_fold(self, f, batch_size):
        ids = np.random.randint(0, len(self.y[f]), size=batch_size).tolist()
        ids.sort()
        X = np.zeros((len(ids), 256, 256, self.c))
        for i in range(len(ids)):
            X[i,:,:,:] = augment(self.X[f][ids[i],:,:,:].reshape((256, 256, self.c))).reshape((1, 256, 256, self.c))
        return (X, self.y[f][ids,:])


# In[367]:

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


# In[368]:

# im = io.imread(os.path.join(PLANET_KAGGLE_ROOT, 'train-tif-v2', 'train_{}.tif'.format(5)))
# plt.imshow((im[:,:,3]) / 6)
# for i in range(2):
#     for j in range(4):
#         im_new = augment(im, (i, j))
#         plt.figure()
#         plt.imshow((im_new[:,:,3]) / 6)


# In[369]:

# g = gen_training_data(2, True)
# t = next(g)


# In[370]:

def amazon_score(y_true, y_pred):
    y_true = y_true > 0.2
    y_pred = y_pred > 0.2

    y_tp = tf.logical_and(y_true, y_pred)
    y_fn = tf.logical_and(y_true, tf.logical_not(y_pred))
    y_fp = tf.logical_and(tf.logical_not(y_true), y_pred)

    tp = tf.reduce_sum(tf.to_float(y_tp))
    fn = tf.reduce_sum(tf.to_float(y_fn))
    fp = tf.reduce_sum(tf.to_float(y_fp))

    p = tf.where(tp + fp > 0, tp / (tp + fp), 0)
    r = tf.where(tp + fn > 0, tp / (tp + fn), 1)
    s = tf.where(p + r > 0, 5 * p * r / (4 * p + r), 0)

    return s


# In[371]:

def new_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(64, 64, 4)))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(N_TAGS, activation='sigmoid'))

    model.compile(metrics=[amazon_score, 'accuracy'],
                  loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001))
    return model

def load_model(path):
    model = new_model()
    model.load_weights(path)
    return model

from resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input
from keras.models import Model

import numpy as np

def new_resnet50(input_shape=(256, 256, 4)):
    base_model = ResNet50(weights=None, include_top=False, pooling='avg', input_tensor = Input(shape=input_shape))

    x = base_model.output
    predictions = Dense(17, activation='sigmoid')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
#     for layer in base_model.layers:
#         layer.trainable = False

#     for layer in model.layers[-24:]:
#         layer.trainable = True

    model.compile(metrics=[amazon_score, 'accuracy'],
                  loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001))
    return model


# In[372]:

def predict(model_path):
    model = load_model(model_path)
    print('Model weights loaded')

    pred = None
    cnt = 0
    print('Start predicting..')
    for X_test in gen_test_data(100):
        y_test = model.predict_on_batch(X_test)
        if pred is None:
            pred = y_test
        else:
            pred = np.concatenate((pred, y_test))
        cnt += len(y_test)
        print('Predicted {} images'.format(cnt))
    print('Predicted all {} images'.format(cnt))

    print('Saving raw predictions...')
    np.save('raw_pred.npy', pred)
    print('Saved')

    result = output(pred)
    print('Saving submission file...')
    result.to_csv('submission.csv', index = None)
    print('Saved')
    return result


# Training


# In[375]:

def train():
    d = Data(tif=False)

    m = new_resnet50(input_shape=(256,256,3))
    # m = new_model()

    h = m.fit_generator(
        d.gen_train(32), steps_per_epoch=8000,
        epochs=40, initial_epoch=0,
        validation_data=d.gen_val(100), validation_steps=80,
        callbacks=[
            ModelCheckpoint('weights-v9.hdf5', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0, verbose=1)],
        max_q_size=10)

    return h


# In[377]:

train()


