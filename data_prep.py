# Import library packages.

# Import relevant libraries

import scipy
from scipy import spatial
from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm
from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as F
from torch.utils.data import Dataset
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import torch as T
from torch import nn, optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import os
import cv2
import PIL
import numpy as np
import torchvision
import glob
from random import shuffle
from PIL import Image
from os import listdir
from numpy import asarray
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imblearn.under_sampling import RandomUnderSampler
import random
from tensorflow.keras.losses import MSE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from numpy import array
from PIL import Image
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

############ We use the easyfsl library mainly for the work ###############

!pip install easyfsl  # Install easyfsl.
from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average

####################### Load the AIDER Dataset ################################

#### load all images in a directory into memory.

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
   return[int(text) if text.isdigit() else text.lower() for text in _nsre.split(s   )]


def load_images(path, size = (1024,1024)):
    data_list = list() # enumerate filenames in directory, assume all are images
    for filename in sorted(os.listdir(path),key=natural_sort_key):
      pixels = load_img(path + filename, target_size = size) # Convert to numpy array.
      pixels = img_to_array(pixels).astype('float32')
      pixels = cv2.resize(pixels,(128,128)) # Resize to 128 x 128.
      pixels = pixels/255
      data_list.append(pixels)
    return asarray(data_list)

# Five different directory corresponding to 5 different classes.

pathtrain_collapsed = '/content/drive/MyDrive/MAML-keras-master/data/AIDER III/Train/1.collapsed/'
pathtrain_fire = '/content/drive/MyDrive/MAML-keras-master/data/AIDER III/Train/2.fire/'
pathtrain_flood = '/content/drive/MyDrive/MAML-keras-master/data/AIDER III/Train/3.flood/'
pathtrain_traffic = '/content/drive/MyDrive/MAML-keras-master/data/AIDER III/Train/4.traffic/'
pathtrain_normal = '/content/drive/MyDrive/MAML-keras-master/data/AIDER III/Train/5.normal/'


data_train_collapsed = load_images(pathtrain_collapsed)
data_train_fire = load_images(pathtrain_fire)
data_train_flood = load_images(pathtrain_flood)
data_train_traffic  = load_images(pathtrain_traffic)
data_train_normal = load_images(pathtrain_normal)
