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

# text = open("txt_sentoken/neg/cv000_29416.txt","r")
# text = text.read()
# print text

neg_reviews = []
pos_reviews = []

for filename in os.listdir("txt_sentoken/neg/"):
    text = open("txt_sentoken/neg/"+filename, "r")
    text = text.read()
    neg_reviews.append(text)

for filename in os.listdir("txt_sentoken/pos/"):
    text = open("txt_sentoken/pos/"+filename, "r")
    text = text.read()
    pos_reviews.append(text)


