
# coding: utf-8



'''ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

#     x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model





















# # Common

# In[171]:

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

# # Matplotlib for visualization
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


# In[172]:

PLANET_KAGGLE_ROOT = '/data/planet-data/'
if not os.path.exists(PLANET_KAGGLE_ROOT):
    PLANET_KAGGLE_ROOT = '/Users/jiayou/Documents/Kaggle Data/Amazon'

N_TAGS = 17
N_TRAIN = 40479
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


# In[173]:

# img = load_train_image(1)
# plt.imshow(img)
# img.shape


# In[174]:

train_labels = pd.read_csv(os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv'))
train_labels


# In[175]:

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


# In[178]:

def get_training_data(file_ids, tif=False, dbg=False, verbose=False):
    if verbose:
        print('Getting {} training images...'.format(len(file_ids)))
    X_train = np.zeros((len(file_ids), 64, 64, 4 if tif else 3))
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


# In[228]:

class Data:
    def __init__(self):
        print('Loading data...')
        self.X = [0] * 5
        self.y = [0] * 5
        for i in range(5):
            self.X[i] = np.load('X.{}.npy'.format(i))
            self.y[i] = np.load('y.{}.npy'.format(i))
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
        X = np.zeros((len(ids), 256, 256, 4))
        for i in range(len(ids)):
            X[i,:,:,:] = augment(self.X[f][i,:,:,:].reshape((256, 256, 4))).reshape((1, 256, 256, 4))
        return (X, self.y[f][ids,:])


# In[222]:

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


# In[227]:

# im = io.imread(os.path.join(PLANET_KAGGLE_ROOT, 'train-tif-v2', 'train_{}.tif'.format(5)))
# plt.imshow((im[:,:,3]) / 6)
# for i in range(2):
#     for j in range(4):
#         im_new = augment(im, (i, j))
#         plt.figure()
#         plt.imshow((im_new[:,:,3]) / 6)


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


# In[231]:

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

# from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input
from keras.models import Model

import numpy as np

def new_resnet50():
    base_model = ResNet50(weights=None, include_top=False, pooling='avg', input_tensor = Input(shape=(256, 256, 4)))
    
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

# In[ ]:

# def get_cv_data(i):
#     X_train_data = []
#     y_train_data = []
#     for j in range(5):
#         if j != i:
#             X_train_data.append(np.load('X.{}.npy'.format(j)))
#             y_train_data.append(np.load('y.{}.npy'.format(j)))
#         else:
#             X_val = np.load('X.{}.npy'.format(j))
#             y_val = np.load('y.{}.npy'.format(j))
                                
#     X_train = np.concatenate(tuple(X_train_data), axis = 0)
#     y_train = np.concatenate(tuple(y_train_data), axis = 0)
#     return X_train, y_train
    


# In[191]:

def train_from_raw():
    X_train, y_train = get_training_data([x for x in range(N_TRAIN) if x % 5 != 0], tif=True, verbose=True)
    gc.collect()
    
    X_val, y_val = get_training_data([x for x in range(N_TRAIN) if x % 5 == 0], tif=True, verbose=True)
    gc.collect()
    
    model = new_model()
    
    h = model.fit(
        X_train, y_train, batch_size=32, verbose=1,
        validation_data=(X_val, y_val),
        epochs=40, initial_epoch=0,
        callbacks=[
            ModelCheckpoint('weights-v7.hdf5', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001, verbose=1)],
    )
    return h


# In[230]:

def train():
    d = Data()
    
    m = new_resnet50()
    
    h = model.fit_generator(
        d.gen_train(32), steps_per_epoch=1000, 
        epochs=40, initial_epoch=0,
        validation_data=d.gen_val(100), validation_steps=80,
        callbacks=[
            ModelCheckpoint(
                'weights-v8.hdf5',
                save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001, verbose=1)],
        max_q_size=10)
    return h


# In[ ]:

train()


# In[12]:

# model = new_model()
# model = load_model('/Users/jiayou/weights-v1.02-0.60-0.10.hdf5')
# model = new_resnet50()


# In[16]:

# h = model.fit_generator(
#         gen_training_data(30), steps_per_epoch=1000, 
#         epochs=30, initial_epoch=0,
#         validation_data=gen_validation_data(100), validation_steps=80,
#         callbacks=[
#             ModelCheckpoint(
#                 'weights-v5.{epoch:02d}-{val_amazon_score:.2f}-{val_loss:.2f}.hdf5',
#                 save_best_only=True, verbose=1),
#             ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, verbose=1)],
#         max_q_size=10)

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


# # In[ ]:




# # In[134]:

# def get_training_file_ids(draw_size):
#     file_ids = np.random.randint(0, N_TRAIN, size=draw_size).tolist()
#     for i in range(len(file_ids)):
#         if file_ids[i] % 5 == 0:
#             file_ids[i] = (file_ids[i] + 1) % N_TRAIN
#     return file_ids


# # In[136]:

# def get_calib_params():
#     draw_size = 1000
#     file_ids = get_training_file_ids(draw_size)
#     ref_color = [[], [], [], []]

#     for i in range(draw_size):
#         current_im = io.imread(os.path.join(PLANET_KAGGLE_ROOT, 'train-tif-v2', 'train_{}.tif'.format(file_ids[i])))
#         flatten_im = current_im.reshape((-1, 4))
#         for j in range(4):
#             ref_color[j] += flatten_im[:, j].tolist()

#     ref_color = np.array(ref_color)
#     ref_param = np.zeros((2, 4))
#     ref_param[0,:] = ref_color.mean(axis = 1)
#     ref_param[1,:] = ref_color.std(axis = 1)
#     return ref_param


# # In[161]:




# # In[220]:

# im = io.imread(os.path.join(PLANET_KAGGLE_ROOT, 'train-tif-v2', 'train_{}.tif'.format(5)))
# im = calib_image(im)
# im = im[:3,:3,:]
# im_rot = np.rot90(im, 1, (0, 1))
# im_mir = np.flip(im_rot, 1)
# im_mir
# # plt.imshow((im[:,:,3] + 3) / 6)
# # plt.hist(np.reshape(im[:,:,0:3], (256*256*3, 1)))


# # In[167]:

# for n in range(1):
#     im = io.imread(os.path.join(PLANET_KAGGLE_ROOT, 'train-tif-v2', 'train_{}.tif'.format(n)))
#     im = calib_image(im)
#     plt.figure()
#     plt.title(n)
#     for i, color in enumerate(['r','g','b','k']):
#         plt.hist(np.reshape(im[:,:,i], (256*256)), bins=30, label=color, color=color, histtype='step', range=[-5, 5])
#     plt.legend()


# # In[70]:

# im = io.imread(os.path.join(PLANET_KAGGLE_ROOT, 'train-tif-v2', 'train_{}.tif'.format(0)))
# im = im.astype('float32')
# im = im / 10000 - 0.5
# np.save(os.path.join(PLANET_KAGGLE_ROOT, 'train_0.tif.npy'), im)


# # In[127]:

# a = np.zeros((4,2,2,2))
# b = a[[1,2],:,:,:]
# b[0,0,:,:] = np.array([1,2,3,4]).reshape((2,2))
# b


# # In[132]:

# [random.randint(0, 3) for i in range(10)]


# # In[ ]:



