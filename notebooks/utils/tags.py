# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_column', 100)

# NumPy for numerical computing
import numpy as np
np.random.seed(123)
import random
random.seed(123)

from sklearn.metrics import roc_curve, auc

from matplotlib import pyplot as plt

import os

PLANET_KAGGLE_ROOT = '/data/planet-data/'
if not os.path.exists(PLANET_KAGGLE_ROOT):
    PLANET_KAGGLE_ROOT = '/Users/jiayou/Documents/Kaggle Data/Amazon'

N_TAGS = 17
N_TRAIN = 40479
N_TEST_T = 40669
N_TEST_F = 20522
N_TEST = N_TEST_T + N_TEST_F

def get_tag_map():
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
    return label_map

_thres = [0.24, 0.22, 0.25, 0.23, 0.26, 0.19, 0.24, 0.23, 0.1, 0.16, 0.14,
         0.2, 0.17, 0.33, 0.04, 0.12, 0.07]

class Tags:
    def __init__(self):
        self.train_tags = pd.read_csv(os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv'))
        self.tag_map = {'agriculture': 2,
            'artisinal_mine': 13,
            'bare_ground': 12,
            'blooming': 14,
            'blow_down': 16,
            'clear': 3,
            'cloudy': 9,
            'conventional_mine': 11,
            'cultivation': 7,
            'habitation': 5,
            'haze': 0,
            'partly_cloudy': 10,
            'primary': 1,
            'road': 6,
            'selective_logging': 15,
            'slash_burn': 8,
            'water': 4}
        self.tag_idx = {v: k for k, v in self.tag_map.items()}

    def idx_to_tag(self, idx):
        return self.tag_idx[idx]

    def y_train(self, ids):
        y_train = np.array([[0. for i in range(N_TAGS)] for j in ids])
        for i in range(len(ids)):
            tags = self.train_tags.tags[ids[i]]
            for tag in tags.split(' '):
                if not tag in self.tag_map:
                    continue
                y_train[i][self.tag_map[tag]] = 1.
        return y_train


    def pred_to_output(self, pred, thres):
        result = pd.DataFrame({
            'image_name':
                ['test_{}'.format(i) for i in range(N_TEST_T)] + ['file_{}'.format(i) for i in range(N_TEST_F)],
            'tags': ['' for i in range(N_TEST)]
        })
        for i in range(len(pred)):
            current_pred = pred[i]
            current_tag = self.pred_to_tags_2(current_pred, thres=thres)
            result.iat[i, 1] = current_tag
        return result

    def pred_to_tags(self, y, thres=[0.2]*N_TAGS):
        # weather_labels = [0, 3, 9, 10]
        weather_labels = []
        maxw = 0
        w = 3
        tags = []
        for i in range(N_TAGS):
            tag = self.tag_idx[i]
            if i in weather_labels:
                if y[i] > maxw:
                    maxw = y[i]
                    w = i
            else:
                if y[i] >= thres[i]:
                    tags.append(tag)
    #     tags.append(label_idx[w])
        return ' '.join(tags)

    def pred_to_tags_2(self, y, thres=[0.2]*N_TAGS):
        weather_labels = [0, 3, 9, 10]
        maxw = 0
        w = 3
        tag_list = []
        for i in range(N_TAGS):
            tag = self.tag_idx[i]
            if i in weather_labels:
                if y[i] > maxw:
                    maxw = y[i]
                    w = i
            else:
                if y[i] >= thres[i]:
                    tag_list.append(tag)
        if maxw >= 0.5:
            for i in weather_labels:
                if y[i] >= 0.5:
                    tag_list.append(self.tag_idx[i])
        else:
            for i in weather_labels:
                if y[i] >= thres[i]:
                    tag_list.append(self.tag_idx[i])
        return ' '.join(tag_list)

    def plot_roc(self, pred, true, title='Receiver Operating Characteristic'):
        fig = plt.figure(figsize=(15,15))
        plt.title(title)

        for i in range(N_TAGS):
            # Plot ROC curve
            fpr, tpr, thresholds = roc_curve(true[:,i], pred[:,i])
            area = auc(fpr, tpr)
            plt.plot(fpr, tpr, label='{} ({:.4f})'.format(self.idx_to_tag(i), area), ls=[ '-' , '--' , '-.' , ':'][random.randint(0, 3)])

        # Diagonal 45 degree line
        plt.plot([0,1],[0,1],'k--')

        # Axes limits and labels
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.1,1.1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc='lower right')
        plt.show()

