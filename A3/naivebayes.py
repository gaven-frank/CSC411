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

#define global variables
k = 1
m = 2

neg_reviews = []
pos_reviews = []

keywords_range = ["no originality","doesn't matter","does not make for", 'clueless']
keyword_freq_neg = {}
keyword_freq_pos = {}

for filename in os.listdir("txt_sentoken/neg/"):
    text = open("txt_sentoken/neg/"+filename, "r")
    text = text.read()
    neg_reviews.append(text)

for filename in os.listdir("txt_sentoken/pos/"):
    text = open("txt_sentoken/pos/"+filename, "r")
    text = text.read()
    pos_reviews.append(text)


for keyword in keywords_range:
    keyword_freq_neg[keyword] = 0
    keyword_freq_pos[keyword] = 0

    for review in neg_reviews:
        if review.find(keyword) != -1:
            keyword_freq_neg[keyword] += 1

    for review in pos_reviews:
        if review.find(keyword) != -1:
            keyword_freq_pos[keyword] += 1


for key in keyword_freq_neg:
    keyword_freq_neg[key] += m*k
    keyword_freq_neg[key] = keyword_freq_neg[key] #should be divided by 1000, but will underflow happen

for key in keyword_freq_pos:
    keyword_freq_pos[key] += m*k
    keyword_freq_pos[key] = keyword_freq_pos[key]

print(keyword_freq_neg)