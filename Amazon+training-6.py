
# coding: utf-8

# # Common

# In[1]:

# print_function for compatibility with Python 3
from __future__ import print_function
print('print function is ready to serve')

# NumPy for numerical computing
import numpy as np
np.random.seed(123)

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_column', 100)

# Matplotlib for visualization
# from matplotlib import pyplot as plt

# display plots in the notebook
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
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean


# In[2]:

# PLANET_KAGGLE_ROOT = '/Users/jiayou/Documents/Kaggle Data/Amazon'
PLANET_KAGGLE_ROOT = '/data/planet-data/'

N_TAGS = 17
N_TRAIN = 40479
# N_USE = 32000
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

def load_train_image(n, dbg=False):
    path = os.path.abspath(os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg', 'train_{}.jpg'.format(n)))
    if os.path.exists(path):
        img = io.imread(path)
        # if dbg:
        #     plt.figure()
        #     plt.imshow(img)
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
      
def preprocess_image(img):
    img = downscale_local_mean(img, (2, 2, 1))
    img = np.array(img)
    img = img.astype('float32')
    img = img / 255 - 0.5
    return img


# In[3]:

# img = load_train_image(1)
# plt.imshow(img)
# img.shape


# In[4]:

train_labels = pd.read_csv(os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv'))
train_labels


# In[5]:

# label_map = {
#     'clear':0,
#     'partly_cloudy':1,
#     'haze':2,
#     'cloudy':3
# }
# label_idx = {v: k for k, v in label_map.items()}

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



# In[14]:

def get_training_data(file_ids, dbg=False):
    X_train = np.array([preprocess_image(load_train_image(fname, dbg)) for fname in file_ids])
    
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


# In[7]:

# g = gen_training_data(2, True)
# t = next(g)


# In[8]:

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


# In[9]:

def new_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(64, 64, 3)))

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

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input
from keras.models import Model

import numpy as np

def new_resnet50():
    base_model = ResNet50(weights=None, include_top=False, input_tensor = Input(shape=(128, 128, 3)))
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = Flatten()(x)
    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
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


# In[10]:

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


# # Training

# In[11]:

# X_train, y_train = get_training_data([
#     10856, 35105, 10421, 15999, 37381, 
#     26157, 25945, 5195, 17380, 18042,
#     28374, 31971, 27748, 19283, 24197,
#     23567, 19779, 17303, 27781, 20267
# ])


# In[12]:

# model = new_model()
# model = load_model('/Users/jiayou/weights-v1.02-0.60-0.10.hdf5')
model = new_resnet50()


# In[16]:

h = model.fit_generator(
        gen_training_data(30), steps_per_epoch=1000, 
        epochs=40, initial_epoch=0,
        validation_data=gen_validation_data(100), validation_steps=80,
        callbacks=[
            ModelCheckpoint(
                'weights-v6.hdf5',
                save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, verbose=1)],
        max_q_size=10)

# h = model.fit_generator(
#         gen_training_data(1), steps_per_epoch=20,
#         epochs=50, initial_epoch=0,
#         validation_data=gen_validation_data(10), validation_steps=1,
#         callbacks=[ModelCheckpoint('weights-v1.{epoch:02d}-{val_amazon_score:.2f}-{val_loss:.2f}.hdf5')],
#         max_q_size=2)

# h = model.fit(
#     X_train, y_train, batch_size=1, epochs=50, verbose=1,
#     validation_data=(X_train, y_train),
#     callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, verbose=1)]
# )

# h = model.fit(X_train, y_train, batch_size=10, epochs=12, verbose=1, initial_epoch=10)


# # Prediction

# In[120]:

# out = predict('/data/kaggle-planet/weights-v4.24-0.86-0.14.hdf5')


# # In[ ]:




# # In[47]:

# pred = np.load(os.path.join(PLANET_KAGGLE_ROOT, 'preds', 'raw_pred.npy'))
# result = output(pred)
# result.to_csv(os.path.join(PLANET_KAGGLE_ROOT, 'preds', 'submission-3.csv'), index = None)


# In[ ]:




# In[ ]:




# In[ ]:



