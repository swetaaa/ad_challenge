# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 15:47:27 2014

@author: Sweta
"""
import numpy as np

class StochasticGradientDescentMethod():
    """ A class for the stochastic gradient descent method"""   
    
    def __init__(self,train_labels,train_features):
        self.label_vec = train_labels
        self.feature_mat = train_features
        self.weights_vec = np.zeros(self.feature_mat.shape[1])
        self.scaling_vec = np.linalg.norm(self.feature_mat,np.inf,axis=0)
        self.feature_mat = self.scale(self.feature_mat)
        
    def scale(self,mat):         
         scaled_mat = (mat.T/self.scaling_vec[:,None]).T
         return scaled_mat
        
    def logistic_loss_function_gradient(self,x,y,p):
        '''
        logistic loss function = log(1+exp(-yp))
        x : features (vector)
        y : label (scalar)
        p : probability = <w,x>, w - weights   
        '''
        l2 = np.exp(-y*p)
        l1 = 1 + l2 # outer derivative of log
        return ( (-1.0) * l2 * y * x) / l1
        
    def learn(self,no_iter,shuffle):
        '''
        note: logistic loss function is used by default
        no_iter : how many times the algorithm is applied to the training set
        shuffle : boolean indicating if training shall be shuffled before each iteration
        '''            
        eta   = 0.001  # learning rate
        theta = 0.7    # regularization parameter

        for it in range(0,no_iter):
            print('iteration : '+str(it))
            if shuffle == True:
                combi_mat =  np.append(self.label_vec[:,None],self.feature_mat,1)
                np.random.shuffle( combi_mat )
                self.feature_mat = combi_mat[:,1:] 
                self.label_vec   = combi_mat[:,0]
            # run SGD for every sample
            for ind, row in enumerate(self.feature_mat):
                p = np.inner(self.weights_vec,row) # compute probability
                self.weights_vec -= eta * theta*self.weights_vec #l2 Regularization
                self.weights_vec -= eta * self.logistic_loss_function_gradient(row,self.label_vec[ind],p)
                
        return self.weights_vec
        
        
    def predict(self,test_id,test_features):
    
        if np.linalg.norm(self.weights_vec) == 0:
            raise ValueError('All weights are zero! (Possible reason: learn() was not called.)')
        # scale feature mat by scaling_vec (the same as used for the training set)
        test_features = self.scale(test_features)   
        return ( np.matrix([ test_id, np.dot(test_features,self.weights_vec) ]) ).T
        
     
        
    
