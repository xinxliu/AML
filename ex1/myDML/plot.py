# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

if  __name__ == '__main__':
    train_file = open('dataset/moons/moons_train.pkl', 'rb')
    test_file = open('dataset/moons/moons_test.pkl', 'rb')
    moons_train = pkl.load(train_file) # tuple of training data
    moons_test = pkl.load(test_file) # tuple of testing data
    train_ins = moons_train[0]
    train_x = train_ins[:,0]
    train_y = train_ins[:,1]
    plt.plot(train_x,train_y,'ro')
   
    
    
