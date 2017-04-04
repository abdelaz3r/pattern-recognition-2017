# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 09:58:25 2017

@author: alvin
"""

#imports
import numpy as np
from sklearn import svm,metrics,cross_validation as CV
import argparse

#load data
train = np.loadtxt('train.csv', dtype=int, delimiter=',') 
train_y = train[:,0:1].flatten()
train= train[:,1:]


def tune(k='linear' , c=1,  g=0.001, v=10):
 train = np.loadtxt('train.csv', dtype=int, delimiter=',') 
 train_y = train[:,0:1].flatten()
 train= train[:,1:]
 clf = svm.SVC(kernel=k, C=c,gamma = g)# take parameter from  command line 
 scores = CV.cross_val_score(clf, train, train_y, cv=v)
 print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))


def prediction(k='linear' , c=1,  g=0.001):
 test = np.loadtxt('test.csv', dtype=int, delimiter=',')
 test_y = test[:,0:1].flatten()
 test= test[:,1:]    
 clf = svm.SVC(kernel=k, C=c,gamma = g)
 predicted = clf.predict(test)
 print("Prediction accuracy: ",metrics.accuracy_score(predicted, test_y))
 print("Confusion matrix:\n%s" % metrics.confusion_matrix(predicted, test_y))

#boillerplate syntax for taking agrs from CMD
def main():

 parser = argparse.ArgumentParser() 
 parser.add_argument('c', nargs='*', help="The C parameter for SVM",default=[5], type = float)
 parser.add_argument('cv', nargs='*',help="number of folds for K-fold CV", default=[5], type = int)
 parser.add_argument('k', type=str, help="the kernel 'rbf','linear','poly',...")
 parser.add_argument('g', help="gamma, The kernel coeff ",type=int)
 
 args = parser.parse_args()

 for c in args.c :
     for v in args.cv:
         tune(args.k,c, args.g,v)# to run prediction change call to prediction

if __name__ == '__main__':
    main()
