import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = ''

train_data = './train_data/'
test_data = './test_data/'

def one_hot_label(img):
    label = img.split('.')[0]

    if label == 'tyskie':
        ohl = np.array([1,0])
    elif label == 'farmstead':
        ohl = np.array([0,1])

    return ohl

def train_data_with_label():
    train_images = []

    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (299, 299))
        train_images.append([np.array(img), one_hot_label(i)])

    shuffle(train_images)
    return train_images

def test_data_with_label():
    test_images = []

    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (299, 299))
        test_images.append([np.array(img), one_hot_label(i)])

    return test_images






