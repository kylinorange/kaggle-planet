
# coding: utf-8

# In[4]:

# print_function for compatibility with Python 3
from __future__ import print_function
print('print function is ready to serve')

# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_column', 100)

# Matplotlib for visualization
from matplotlib import pyplot as plt

# Seaborn for easier visualization
import seaborn as sns

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Import RandomForestClassifier and GradientBoostingClassifer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Function for splitting training and test set
from sklearn.model_selection import train_test_split

# Function for creating model pipelines
from sklearn.pipeline import make_pipeline

# For standardization
from sklearn.preprocessing import StandardScaler

# Helper for cross-validation
from sklearn.model_selection import GridSearchCV

# Classification metrics (added later)
from sklearn.metrics import roc_curve, auc


# In[17]:

import os
from skimage import io

PLANET_KAGGLE_ROOT = '/Users/jiayou/Documents/Kaggle Data/Amazon'

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

def load_train_image(n, dbg=False):
    path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg', 'train_{}.jpg'.format(n)))
    if os.path.exists(path):
        img = io.imread(path)
        if dbg:
            plt.figure()
            plt.imshow(img)
        return img
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))

def load_test_image(n):
    path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, 'test-jpg', 'test_{}.jpg'.format(n)))
    if os.path.exists(path):
        return io.imread(path)
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))
        


# In[15]:

# img = load_train_image(1)
# plt.imshow(img)
# img.shape


# In[10]:

train_labels = pd.read_csv(os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv'))
train_labels


# In[11]:

# Build list with unique labels
label_list = []
for tag_str in train_labels.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)

label_map = {}
for i in range(len(label_list)):
    label_map[label_list[i]] = i
label_map


# In[2]:

def get_training_data(file_ids, dbg=False):
    X_train = np.array([np.array(load_train_image(fname, dbg)) for fname in file_ids])
    X_train = X_train.astype('float32')
    X_train /= 255
    
#     file_names = ['train_{}'.format(n) for n in file_ids]
#     train_labels = pd.read_csv(os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv'))
#     cond = train_labels.image_name.apply(lambda x: True if x in file_names else False).tolist()
#     train_labels = train_labels[cond]
    
    y_train = np.array([[0. for i in range(17)] for j in file_ids])
    for i in range(len(file_ids)):
        tags = train_labels.tags[file_ids[i]]
        if dbg:
            print(file_ids[i], tags)
        for tag in tags.split(' '):
            y_train[i][label_map[tag]] = 1.
    if dbg:
        print(y_train)
    
    return (X_train, y_train)

N_TRAIN = 40479
N_USE = 32000
# N_TRAIN = 20
# N_USE = 10

def gen_training_data(batch_size, dbg=False):
    while 1:
        file_ids = np.random.randint(0, N_USE, size=batch_size).tolist()
        file_ids.sort()
        if dbg:
            print('file ids: ', file_ids)
        yield get_training_data(file_ids, dbg)


def gen_validation_data(batch_size):
    while 1:
        file_ids = np.random.randint(N_USE, N_TRAIN, size=batch_size).tolist()
        file_ids.sort()
        yield get_training_data(file_ids)


# In[19]:

# g = gen_training_data(2, True)
# t = next(g)


# In[6]:

import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# In[7]:

import tensorflow as tf

def amazon_score(y_true, y_pred):
    y_true = y_true > 0.5
    y_pred = y_pred > 0.5
    
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


# In[118]:

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256,256,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='relu'))

model.compile(metrics=[amazon_score],
              loss='mean_absolute_error',
              optimizer='sgd')


# In[113]:

# model.fit(X_train, y_train, batch_size=10, epochs=10, verbose=1)
# h = model.fit(X_train, y_train, batch_size=10, epochs=12, verbose=1, initial_epoch=10)


# In[119]:

h = model.fit_generator(
        gen_training_data(30), steps_per_epoch=1000, 
        epochs=3, initial_epoch=0,
        validation_data=gen_validation_data(100), validation_steps=80,
        callbacks=[ModelCheckpoint('weights-v1.{epoch:02d}-{val_amazon_score:.2f}-{val_loss:.2f}.hdf5')],
        max_q_size=2)

# h = model.fit_generator(
#         gen_training_data(10), steps_per_epoch=10, 
#         epochs=5, initial_epoch=0,
#         validation_data=gen_validation_data(10), validation_steps=1,
#         callbacks=[ModelCheckpoint('weights-v1.{epoch:02d}-{val_amazon_score:.2f}-{val_loss:.2f}.hdf5')],
#         max_q_size=2)


# In[20]:

# prediction = model.predict(X_test)
# np.floor(prediction + 0.5).astype('int32')


# In[21]:

# y_test.astype('int32')


# In[22]:

# model.evaluate(X_test, y_test)


# In[23]:

# model.model.history.history


# - Net design
# - Use all training data
# - Only one weather

# In[ ]:



