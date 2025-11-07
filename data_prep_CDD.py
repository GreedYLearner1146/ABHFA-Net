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
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

############ We use the easyfsl library mainly for the work ###############

!pip install easyfsl  # Install easyfsl.
from easyfsl.samplers import TaskSampler
from easyfsl.utils import plot_images, sliding_average
from PIL import ImageFile, Image
#### load all images in a directory into memory.
import re

####################### Meta-Train class loading ##########################

ImageFile.LOAD_TRUNCATED_IMAGES = True

def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
   return[int(text) if text.isdigit() else text.lower() for text in _nsre.split(s   )]


def load_images(path, size = (224,224)):
    data_list = list()# enumerate filenames in directory, assume all are images
    for filename in sorted(os.listdir(path),key=natural_sort_key):
      pixels = load_img(path + filename, target_size = size)# Convert to numpy array.
      pixels = img_to_array(pixels).astype('float32')
      pixels = cv2.resize(pixels,(128,128))# Need to resize images first, otherwise RAM will run out of space.
      pixels = pixels/255
      #pixels = cv2.threshold(pixels, 128, 128, cv2.THRESH_BINARY)
      data_list.append(pixels)
    return asarray(data_list)


path_infra = 'Insert your CDD directory here'
path_wild_fire = 'Insert your CDD directory here'
path_landslide = 'Insert your CDD directory here'
path_ND_street = 'Insert your CDD directory here'
path_ND_sea = 'Insert your CDD directory here'
path_ND_human = 'Insert your CDD directory here'


data_train_infra = load_images(path_infra)
data_wild_fire = load_images(path_wild_fire)
data_landslide = load_images(path_landslide)
data_ND_street = load_images(path_ND_street)
data_ND_sea  = load_images(path_ND_sea)
data_test_ND_human = load_images(path_ND_human)

##################################################

# Prepare array for storing images and labels.

imgarrayinfra =  []
labelarrayinfra = []

imgarraywildfire =  []
labelarraywildfire = []

imgarraylandslide =  []
labelarraylandslide = []

imgarraystreet =  []
labelarraystreet = []

imgarraysea =  []
labelarraysea = []


imgarrayhuman =  []
labelarrayhuman = []


################################################################

for b in data_train_infra:
    imgarrayinfra.append(b)
    labelarrayinfra.append(0)

################################################################

for d in data_wild_fire:
   imgarraywildfire.append(d)
   labelarraywildfire.append(1)

################################################################

for g in data_landslide:
    imgarraylandslide.append(g)
    labelarraylandslide.append(2)

################################################################

for j in data_ND_street:
    imgarraystreet.append(j)
    labelarraystreet.append(3)

for k in data_ND_sea:
    imgarraysea.append(k)
    labelarraysea.append(4)

for h in data_test_ND_human:
    imgarrayhuman.append(h)
    labelarrayhuman.append(5)

####################### Meta-Test class loading ####################

path_ND_forest = 'Insert your CDD directory here'
path_water = 'Insert your CDD directory here'
path_urban_fire = 'Insert your CDD directory here'
path_drought = 'Insert your CDD directory here'
path_earthquake = 'Insert your CDD directory here'


data_ND_forest = load_images(path_ND_forest)
data_test_earthquake = load_images(path_earthquake)
data_test_water = load_images(path_water)
data_test_urban_fire = load_images(path_urban_fire)
data_test_drought = load_images(path_drought)

###### Prepare array for storing images and labels. ########

imgarrayearthquake =  []
labelarrayearthquake = []

imgarrayurbanfire =  []
labelarrayurbanfire = []

imgarraywater =  []
labelarraywater  = []

imgarraydrought =  []
labelarraydrought = []

imgarrayforest =  []
labelarrayforest = []


for i in data_ND_forest:
    imgarrayforest.append(i)
    labelarrayforest.append(6)

for a in data_test_earthquake:
    imgarrayearthquake.append(a)
    labelarrayearthquake.append(7)

for c in data_test_urban_fire:
    imgarrayurbanfire.append(c)
    labelarrayurbanfire.append(8)

for e in data_test_water:
    imgarraywater.append(e)
    labelarraywater.append(9)

for f in data_test_drought:
    imgarraydrought.append(f)
    labelarraydrought.append(10)

########### meta-train and meta-test array and labels #############
################ meta-train class arrays and labels ###############

trainarray = []
trainlabel = []

testarray =  []
testlabel =  []

for a,b in zip(imgarrayinfra,labelarrayinfra):
  trainarray.append(a)
  trainlabel.append(b)

for c,d in zip(imgarraywildfire,labelarraywildfire):
  trainarray.append(c)
  trainlabel.append(d)

for e,f in zip(imgarraylandslide,labelarraylandslide):
  trainarray.append(e)
  trainlabel.append(f)

for i,j in zip(imgarrayforest,labelarrayforest):
  testarray.append(i)
  testlabel.append(j)

for k,l in zip(imgarraysea,labelarraysea):
  trainarray.append(k)  
  trainlabel.append(l)  

for u,v in zip(imgarrayhuman,labelarrayhuman):
  trainarray.append(u)
  trainlabel.append(v)

################ meta-test class arrays and labels ###############

for m,n in zip(imgarrayearthquake,labelarrayearthquake):
  testarray.append(m)
  testlabel.append(n)

for o,p in zip(imgarrayurbanfire,labelarrayurbanfire):
  testarray.append(o)
  testlabel.append(p)

for q,r in zip(imgarraywater,labelarraywater):
  testarray.append(q)   
  testlabel.append(r)   

for s,t in zip(imgarraydrought,labelarraydrought):
  testarray.append(s)
  testlabel.append(t)

for g,h in zip(imgarraystreet,labelarraystreet):
  trainarray.append(g)
  trainlabel.append(h)

############ Check array shape for meta-train and meta-test class #########################

trainarray_CDD = trainarray
trainlabel_CDD = trainlabel
testarrayCDD = testarray
testlabelCDD = testlabel

print(np.shape(trainarray))
print(np.shape(trainlabel))

print(np.shape(testarray))
print(np.shape(testlabel))
