from pylab import *
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from matplotlib import cm
import os
from scipy.ndimage import filters
import urllib
import string

#define global variables
k = 1
m = 2

neg_reviews_train = []
pos_reviews_train = []
neg_reviews_valid = []
pos_reviews_valid = []
neg_reviews_test = []
pos_reviews_test = []

keywords_range = ["no originality","doesn't matter","does not make for", 'clueless']
keyword_freq_neg = {}
keyword_freq_pos = {}

i = 0
for filename in os.listdir("txt_sentoken/neg/"):
    text = open("txt_sentoken/neg/"+filename, "r")
    text = text.read()
    text = text.lower()
    text = text.translate(None, string.punctuation)
    if i <= 600:
        neg_reviews_train.append(text)
    elif i<=800:
        neg_reviews_valid.append(text)
    else:
        neg_reviews_test.append(text)
    i += 1

i = 0
for filename in os.listdir("txt_sentoken/pos/"):
    text = open("txt_sentoken/pos/"+filename, "r")
    text = text.read()
    text = text.lower()
    text = text.translate(None, string.punctuation)
    pos_reviews_train.append(text)
    if i <= 600:
        pos_reviews_train.append(text)
    elif i<=800:
        pos_reviews_valid.append(text)
    else:
        pos_reviews_test.append(text)
    i += 1


for keyword in keywords_range:
    keyword_freq_neg[keyword] = 0
    keyword_freq_pos[keyword] = 0

    for review in neg_reviews_train:
        if review.find(keyword) != -1:
            keyword_freq_neg[keyword] += 1

    for review in pos_reviews_train:
        if review.find(keyword) != -1:
            keyword_freq_pos[keyword] += 1


for key in keyword_freq_neg:
    keyword_freq_neg[key] += m*k
    keyword_freq_neg[key] = keyword_freq_neg[key] #should be divided by 1000, but will underflow happen

for key in keyword_freq_pos:
    keyword_freq_pos[key] += m*k
    keyword_freq_pos[key] = keyword_freq_pos[key]

print(keyword_freq_neg)