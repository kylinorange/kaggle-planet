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

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3

from resnet50 import ResNet50
from leaky_resnet50 import LeakyResNet50

class Models:

    @staticmethod
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

    @staticmethod
    def new_resnet50(input_shape=(256, 256, 3), leaky=False):
        base_model = None
        if leaky:
            base_model = LeakyResNet50(weights=None, include_top=False, pooling='avg', input_tensor = Input(shape=input_shape))
        else:
            base_model = ResNet50(weights=None, include_top=False, pooling='avg', input_tensor = Input(shape=input_shape))

        x = base_model.output
        predictions = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        # first: train only the top layers (which were randomly initialized)
    #     for layer in base_model.layers:
    #         layer.trainable = False

    #     for layer in model.layers[-24:]:
    #         layer.trainable = True

        model.compile(metrics=[Models.amazon_score, 'accuracy'],
                      loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001))
        return model

    @staticmethod
    def load_resnet50(path, input_shape=(256, 256, 3), leaky=False):
        model = Models.new_resnet50(input_shape, leaky)
        model.load_weights(path)
        return model

    @staticmethod
    def new_incnet(input_shape=(256, 256, 3)):
        base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_tensor = Input(shape=input_shape))

        x = base_model.output
        predictions = Dense(17, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=predictions)

        model.compile(metrics=[Models.amazon_score, 'accuracy'],
                      loss='binary_crossentropy',
                      optimizer=Adam(lr=0.0001))
        return model

    @staticmethod
    def load_incnet(path, input_shape=(256, 256, 3)):
        model = Models.new_incnet(input_shape)
        model.load_weights(path)
        return model

    @staticmethod
    def new_bob():
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

        model.compile(metrics=[Models.amazon_score, 'accuracy'],
                      loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001))
        return model

    @staticmethod
    def load_bob(path):
        model = new_bob()
        model.load_weights(path)
        return model