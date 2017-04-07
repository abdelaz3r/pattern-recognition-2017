# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 09:58:25 2017

@author: alvin
"""

#imports
import numpy as np
from sklearn import svm,metrics,preprocessing
from sklearn.model_selection import GridSearchCV as GS
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

#load data
train = np.loadtxt('C:/Users/alvin/Downloads/train.csv', dtype=float, delimiter=',')
test = np.loadtxt('C:/Users/alvin/Downloads/test.csv', dtype=float, delimiter=',')



train_y = train[:,0:1].flatten().astype(int)
train= train[:,1:]


test_y = test[:,0:1].flatten().astype(int)
test= test[:,1:] 

#scale the data[0,1]. using min max scaler that preserves sparcity 
min_max_scaler = preprocessing.MinMaxScaler()
#fit and transform the scaler with train
train= min_max_scaler.fit_transform(train)
test =min_max_scaler.transform(test) 

  


#find the best parameters by using a grid search CV
# all posibble parameters too long time.
#==============================================================================
# C_range = np.logspace(-10, 0, 5)
# gamma_range = np.logspace(-5, 3, 8)
# k = ['rbf','poly','linear']
# degree = np.array([3,4])
# 
# param_grid = dict(kernel=k, gamma=gamma_range, C=C_range)
# grid = GS(svm.SVC(), param_grid=param_grid,n_jobs=-1 ).fit(train, train_y)
#==============================================================================

# find the best parameters for Linear kernel using LinearSVC

clf = GS(svm.LinearSVC(),dict( C=np.logspace(-10, 0, 30)),n_jobs=-1).fit(train, train_y)
bestpar = clf.best_params_['C']
#print cross validation details
print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
   print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
print("Best settng is %r" %clf.best_params_)

#Now for rbf
#==============================================================================
# grid = GS(svm.SVC(cache_size = 600), param_grid=dict( gamma=np.logspace(-5, 0, 5),\
#                   C=[bestpar]),n_jobs=-1 ).fit(train, train_y)
# 
# #takes too long time 
# #apply PCA
#==============================================================================
pca = PCA(n_components=64)
pca.fit(train)
p_train = pca.fit_transform(train)

#test random withh high gamma and low C
grid = GS(svm.SVC(cache_size = 600), param_grid=dict( gamma=[1.0],\
          C=[0.01,0.001,1]),n_jobs=-1 ).fit(p_train, train_y)

#test with high C low gamma 
grid = GS(svm.SVC(cache_size = 600), param_grid=dict( gamma=[0.1,0.1000001], \
          C=[1.0]),n_jobs=-1 ).fit(p_train, train_y)
#2oom-in on range
grid = GS(svm.SVC(cache_size = 600), param_grid=dict( \
          gamma=np.logspace(-2,-1,12,endpoint=True), C=[1.0]),n_jobs=-1 )\
          .fit(p_train, train_y)

 
#predict
clf = svm.SVC(C=grid.best_params_['C'],gamma=grid.best_params_['gamma'],\
              cache_size=600).fit(train,train)
predicted = clf.predict(test)
print("Prediction accuracy: %0.2f" % (metrics.accuracy_score(predicted, test_y)))
 print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(test_y, predicted)))
