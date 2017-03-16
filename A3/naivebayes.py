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
import operator

#define global variables

neg_reviews_train = []
pos_reviews_train = []
neg_reviews_valid = []
pos_reviews_valid = []
neg_reviews_test = []
pos_reviews_test = []



keyword_range = {}



def load_data():
    global neg_reviews_train
    global pos_reviews_train
    global neg_reviews_valid
    global pos_reviews_valid
    global neg_reviews_test
    global pos_reviews_test
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


def collect_keywords():
    global keyword_range
    global neg_reviews_train
    global pos_reviews_train
    global neg_reviews_valid
    global pos_reviews_valid
    global neg_reviews_test
    global pos_reviews_test
    temp_list = neg_reviews_train+pos_reviews_train+neg_reviews_valid+pos_reviews_valid+neg_reviews_test+pos_reviews_test
    count = 0
    for review in temp_list:
        temp_words = review.split()
        for word in temp_words:
            if word not in keyword_range:
                keyword_range[word] = count
                count += 1

    keyword_range = sorted(keyword_range.items(), key = operator.itemgetter(1))
    return count

def count_occurrence(count):
    global keyword_range
    global neg_reviews_train
    global pos_reviews_train

    keywords_count = np.zeros((count,2))

    for (keyword,index) in keyword_range:

        for review in neg_reviews_train:
            #temp_words = review.split()
            if review.find(keyword) != -1:

                keywords_count[index,0]+=(1.0/review.count(' '))

        for review in pos_reviews_train:
            #temp_words = review.split()
            if review.find(keyword) != -1:

                keywords_count[index,1]+=(1.0/review.count(' '))

        print(index)

    np.save('count_data',keywords_count)

def calculate_prob(m,k):
    global neg_reviews_train
    global pos_reviews_train

    count_data = np.load('count_data.npy')
    print(count_data)
    count_data += m*k
    count_data[:,0] = count_data[:,0]/(len(neg_reviews_train)+k)
    count_data[:,1] = count_data[:,1] / (len(pos_reviews_train) + k)

    print(count_data)



load_data()
count = collect_keywords()
# count_occurrence(count)
calculate_prob(2,0.01)













# for keyword in keywords_range:
#     keyword_freq_neg[keyword] = 0
#     keyword_freq_pos[keyword] = 0
#
#     for review in neg_reviews_train:
#         if review.find(keyword) != -1:
#             keyword_freq_neg[keyword] += 1
#
#     for review in pos_reviews_train:
#         if review.find(keyword) != -1:
#             keyword_freq_pos[keyword] += 1
#
#
# for key in keyword_freq_neg:
#     keyword_freq_neg[key] += m*k
#     keyword_freq_neg[key] = keyword_freq_neg[key] #should be divided by 1000, but will underflow happen
#
# for key in keyword_freq_pos:
#     keyword_freq_pos[key] += m*k
#     keyword_freq_pos[key] = keyword_freq_pos[key]
#
# print(keyword_freq_neg)