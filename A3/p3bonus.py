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
import math
import re
from matplotlib.legend_handler import HandlerLine2D
from scipy.spatial import distance

#define global variables

neg_reviews_train = []
pos_reviews_train = []
neg_reviews_valid = []
pos_reviews_valid = []
neg_reviews_test = []
pos_reviews_test = []

keyword_range = {}

def load_data(num_train = 80,num_valid = 10,num_test = 10):
    global neg_reviews_train
    global pos_reviews_train
    global neg_reviews_valid
    global pos_reviews_valid
    global neg_reviews_test
    global pos_reviews_test
    
    # Generate positive examples
    i = 0
    while 1:
        if i % 2 == 1:
            example = 'a'
        else:
            example = 'b'
    	if i < num_train:
    		pos_reviews_train.append(example)
    	elif i < num_train + num_valid:
    		pos_reviews_valid.append(example)
    	elif i < num_train + num_valid + num_test:
    		pos_reviews_test.append(example)
    	else:
    		break
    	i += 1

    # Generate negative examples
    i = 0
    while 1:
        if i % 2 == 1:
            example = 'a b'
        else:
            example = 'a b'
    	if i < num_train:
    		neg_reviews_train.append(example)
    	elif i < num_train + num_valid:
    		neg_reviews_valid.append(example)
    	elif i < num_train + num_valid + num_test:
    		neg_reviews_test.append(example)
    	else:
    		break
    	i += 1

def collect_keywords():
    global keyword_range
    global neg_reviews_train
    global pos_reviews_train
    global neg_reviews_valid
    global pos_reviews_valid
    global neg_reviews_test
    global pos_reviews_test
    keyword_range = {}
    temp_list = neg_reviews_train+pos_reviews_train+neg_reviews_valid+pos_reviews_valid+neg_reviews_test+pos_reviews_test
    count = 0
    for review in temp_list:
        temp_words = review.split()
        for word in temp_words:
            if word not in keyword_range:
                keyword_range[word] = count
                count += 1
    dict_keyword_range = keyword_range
    keyword_range = sorted(keyword_range.items(), key = operator.itemgetter(1))
    print dict_keyword_range
    return dict_keyword_range , count

def count_occurrence(count):
    global keyword_range
    global neg_reviews_train
    global pos_reviews_train

    keywords_count = np.zeros((count,2))

    for (keyword,index) in keyword_range:

        for review in neg_reviews_train:
            #temp_words = review.split()
            if review.find(keyword) != -1:

                keywords_count[index,0]+=(1.0/len(review.split()))

        for review in pos_reviews_train:
            #temp_words = review.split()
            if review.find(keyword) != -1:

                keywords_count[index,1]+=(1.0/len(review.split()))

        #print(index)
    return keywords_count

def calculate_prob(m,k,count_data):
    count_data += m*k
    count_data[:,0] = count_data[:,0] / (80 + k)
    count_data[:,1] = count_data[:,1] / (80 + k)


    return count_data

def argmax_class(count_data,review,keyword_range):
    sum_pos = 0.0
    sum_neg = 0.0
    keywords = review.split()
    for keyword in keywords:
        sum_neg+=math.log(count_data[keyword_range[keyword],0])
        sum_pos+=math.log(count_data[keyword_range[keyword],1])
    #print(sum_pos,sum_neg)
    # if abs(sum_pos - sum_neg) > 20:
    return(sum_pos,sum_neg)

def generate_dataset(count_of_words,pos_reviews,neg_reviews,dict):
    X = None
    Y = np.zeros((count_of_words,1))
    Y[0:len(pos_reviews)].fill(1)
    #print Y
    count = 0
    for review in pos_reviews:
        word_list = review.split()
        word_vec = np.zeros(count_of_words)
        for word in word_list:
            word_vec[dict[word]] = 1
        if X is None:
            X = word_vec.copy()
        else:
            X = np.vstack((X,word_vec))
        count +=1
        #print count
    count = 0
    for review in neg_reviews:
        word_list = review.split()
        word_vec = np.zeros(count_of_words)
        for word in word_list:
            word_vec[dict[word]] = 1
        if X is None:
            X = word_vec.copy()
        else:
            X = np.vstack((X,word_vec))
        count +=1
        #print count
    return(X,Y)



def bayes():
    global neg_reviews_train
    global pos_reviews_train
    global neg_reviews_valid
    global pos_reviews_valid
    global neg_reviews_test
    global pos_reviews_test
    load_data()
    #print neg_reviews_train
    dict_keyword_range, count = collect_keywords()
    print count
    count_data = count_occurrence(count)
    m = 0.001
    k = 0.009
    Y = np.zeros((20,2))
    Y[:10,1].fill(1)
    Y[10:20,0].fill(1)
    X = np.zeros((20,2))
    # for m in frange(0.009):
    #     for k in frange(0.001,0.05,0.001):
    #X = np.zeros((200,2))
    #print(m,k)    
    accuracy_pos = 0
    accuracy_neg = 0
    #print count_data
    count_data_ = count_data.copy()
    count_data_ = calculate_prob(m, k,count_data_)
    #print count_data
    counter = 0
    for review in neg_reviews_valid:
        (X[counter,0],X[counter,1]) =  argmax_class(count_data_,review,dict_keyword_range)
        counter +=1
    for review in pos_reviews_valid:
        (X[counter,0],X[counter,1]) = argmax_class(count_data_, review,dict_keyword_range)
        counter +=1
    X[:,0] += np.average(X[:,1]) - np.average(X[:,0])
    #print X
    prediction = np.zeros_like(X)
    prediction[np.arange(len(X)),X.argmax(1)] = 1
    #print prediction
    accuracy_pos = np.sum(np.logical_and(prediction[10:20,:],Y[10:20,:]))
    accuracy_neg = np.sum(np.logical_and(prediction[0:10,:],Y[0:10,:]))
    accuracy_neg = accuracy_neg/10.0
    accuracy_pos = accuracy_pos/10.0
    print('Test set accuracy: pos: {}, neg: {}, overall: {}'.format(accuracy_pos,accuracy_neg,(accuracy_neg+accuracy_pos)/2))

    X = np.zeros((20,2))
    accuracy_pos = 0
    accuracy_neg = 0
    #print count_data
    count_data_ = count_data.copy()
    count_data_ = calculate_prob(m, k,count_data_)
    #print count_data
    counter = 0
    for review in neg_reviews_test:
        (X[counter,0],X[counter,1]) =  argmax_class(count_data_,review,dict_keyword_range)
        counter +=1
    for review in pos_reviews_test:
        (X[counter,0],X[counter,1]) = argmax_class(count_data_, review,dict_keyword_range)
        counter +=1
    X[:,0] += np.average(X[:,1]) - np.average(X[:,0])
    #print X
    prediction = np.zeros_like(X)
    prediction[np.arange(len(X)),X.argmax(1)] = 1
    #print prediction
    accuracy_pos = np.sum(np.logical_and(prediction[10:20,:],Y[10:20,:]))
    accuracy_neg = np.sum(np.logical_and(prediction[0:10,:],Y[0:10,:]))
    accuracy_neg = accuracy_neg/10.0
    accuracy_pos = accuracy_pos/10.0
    print('Validation set accuracy: pos: {}, neg: {}, overall: {}'.format(accuracy_pos,accuracy_neg,(accuracy_neg+accuracy_pos)/2))
    return dict_keyword_range,count_data_

def logistic():
    global pos_reviews_train
    global neg_reviews_train
    global pos_reviews_valid
    global neg_reviews_valid
    global pos_reviews_test
    global neg_reviews_test
    dict_keyword_range, count = collect_keywords()
    (X,Y) = generate_dataset(2,pos_reviews_train,neg_reviews_train,dict_keyword_range) 
    X = np.insert(X,0,1,axis = 1)
    (X_valid,Y_valid) = generate_dataset(2,pos_reviews_valid,neg_reviews_valid,dict_keyword_range)
    X_valid = np.insert(X_valid,0,1,axis = 1)
    (X_test,Y_test) = generate_dataset(2,pos_reviews_test,neg_reviews_test,dict_keyword_range)
    X_test = np.insert(X_test,0,1,axis = 1)
    Y = np.zeros((160,2))
    Y[0:80,0].fill(1)
    Y[80:160,1].fill(1)
    Y_valid = np.zeros((20,2))
    Y_valid[0:10,0].fill(1)
    Y_valid[10:20,1].fill(1)
    Y_test = np.zeros((20,2))
    Y_test[0:10,0].fill(1)
    Y_test[10:20,1].fill(1)
    W = np.random.normal(0., 1e-5, size=(X.shape[1],Y.shape[1]))
    train_model_logistic_regression(W,X,Y,X_valid,Y_valid,X_test,Y_test,decay_rate = 0.999)
    return W

def train_model_logistic_regression(W , X , Y, X_valid, Y_valid, X_test, Y_test, num_of_epoch = 1000,decay_rate = 0.995, learning_rate = 1e-2, report_every = 10):
    x_axis = np.linspace(0,num_of_epoch,num_of_epoch/report_every)
    y1 = np.zeros(x_axis.shape[0])
    y2 = np.zeros(x_axis.shape[0])
    y3 = np.zeros(x_axis.shape[0])
    for ep in range(num_of_epoch):
        (O,P) = forward(X,W)
        W = W * decay_rate - learning_rate*df_logistic_regression(X,O,P,Y)
        if ep % report_every == 0:
            #(O,P) = forward(X,W)
            train_acc = 1-(calc_accuracy(P,Y))
            train_cost = cost_function(X,W,Y)

            (O_valid,P_valid) = forward(X_valid,W)
            valid_acc = 1-calc_accuracy(P_valid,Y_valid)
            valid_cost = cost_function(X_valid,W,Y_valid)

            (O_test,P_test) = forward(X_test,W)
            test_acc = 1-calc_accuracy(P_test,Y_test)
            test_cost = cost_function(X_test,W,Y_test)

            y1[ep/report_every] = train_acc
            y2[ep/report_every] = test_acc
            y3[ep/report_every] = valid_acc
            print W
            print 'Epoch {}; train_acc={:1.5f}, train_cost={:1.5f}, test_acc={:1.5f}, test_cost={:1.5f}, valid_acc = {:1.5f}'.format(ep,train_acc,train_cost,test_acc,test_cost,valid_acc)   
    training_set_accuracy, = plt.plot(x_axis, y1, 'r-',label = 'training set error rate')
    test_set_accuracy, = plt.plot(x_axis, y2, 'b-',label = 'test set error rate')
    valid_set_accuracy, = plt.plot(x_axis, y3, 'b-',label = 'validation set error rate')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(handler_map={training_set_accuracy: HandlerLine2D(numpoints=4)})
    plt.title('Multinomial Logistic Regression')
    plt.show()

    return W

def cost_function(X,W,Y):
    (O,P) = forward(X,W)
    return -sum((Y*log(P)))/X.shape[0]    

def forward(X, W):
    O = np.dot(X,W)
    P = softmax(O)
    return (O,P)

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    y = y.T
    return (exp(y)/tile(sum(exp(y),0), (len(y),1))).T

    
def df_logistic_regression(X,O,P,y):
    # print X.shape
    # print O.shape
    # print P.shape
    # print y.shape
    return np.dot((X.T),(P-y))/X.shape[0]
    
def calc_accuracy(P,y):
    P = (P == P.max(axis=1)[:,None]).astype(int)
    count = 0
    for j in range(P.shape[0]):
        flag = 1
        for i in range(P.shape[1]):
            if P[j,i] != y[j,i]:
                flag = 0
        count += flag
    return float(count)/y.shape[0]

def part9():
    dict_keyword_range,count_data_ = bayes()
    W = logistic()
    print count_data_

random.seed(666)
part9()
