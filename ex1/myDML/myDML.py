# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 24:00:00 2017

@author: Zhiyu
"""
# import your module here
import numpy as np
import time
from threading import Thread
import functools

# (global) variable definition here
TRAINING_TIME_LIMIT = 60*10
A = 0

def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

@timeout(TRAINING_TIME_LIMIT)
def train(traindata,lr=0.005,max_iter = 100000,epsilon = 0.0001):
    instance = traindata[0]
    ins_label = traindata[1]
    global A
    max_iter = 100*instance.shape[0]
    print('max_iter: %d' % max_iter)
    A = np.eye(instance.shape[1])
 #   A = np.random.normal(size=(instance.shape[1],instance.shape[1]))
    P_MAT = np.ones(instance.shape[0])
    label = ins_label[:,None]==ins_label[None]
    for iteration in range(max_iter):
        i = iteration%instance.shape[0]
        instance_i = instance - instance[i] #(x_j-x_i)^T
        instance_i = np.dot(instance_i,A.T) #(A(x_j-x_i))^T
        P_MAT = (np.linalg.norm(instance_i,axis=1))**2 #(200,) 
        P_MAT = np.exp(-P_MAT)
        P_MAT[i] = 0 #p_ii=0
        P_MAT = P_MAT/np.sum(P_MAT)
        P_MAT=P_MAT.reshape(instance.shape[0],1)
        outer_mat_1 = P_MAT[label[i]].sum()*np.dot((P_MAT*instance_i).T,instance_i)
        outer_mat_2 = np.dot((P_MAT[label[i]]*instance_i[label[i]]).T,instance_i[label[i]])
        grad = 2*np.dot(A,outer_mat_1-outer_mat_2)
        A += lr*grad
        if iteration%1000 == 0:
            print("iter: %d, grad:%f" %(iteration,np.linalg.norm(grad)))
    return 0

def Euclidean_distance(inst_a, inst_b):
    return np.linalg.norm(inst_a - inst_b)

def distance(inst_a, inst_b):
    dist = np.linalg.norm(np.dot(A,inst_a)-np.dot(A,inst_b))  
    return dist

# main program here
if  __name__ == '__main__':
    pass