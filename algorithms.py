# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 15:47:27 2014

@author: Sweta
"""
import numpy as np
import scipy as sp

class StochasticGradientDescentMethod():
    """ A class for the stochastic gradient descent method"""   
    
    def __init__(self,train_labels=np.zeros((1,1)),train_features=np.zeros((1,1))):
        self.label_vec = train_labels
        self.feature_mat = train_features
        self.weights_vec = np.zeros(self.feature_mat.shape[1])
        self.scaling_vec = np.linalg.norm(self.feature_mat,np.inf,axis=0)
        self.feature_mat = self.scale(self.feature_mat)
        self.iter = 1
        self.train_size = self.label_vec.shape[0]
        
    def setTrainData(self,train_labels,train_features,init=False):
        self.label_vec = train_labels
        self.feature_mat = train_features
        if init:
            self.weights_vec = np.zeros(self.feature_mat.shape[1])
            self.scaling_vec = np.linalg.norm(self.feature_mat,np.inf,axis=0)
            if self.train_size == 1:
                self.train_size = self.label_vec.shape[0]
        self.feature_mat = self.scale(self.feature_mat)
        
    def setTrainSize(self,train_size):
        self.train_size = train_size
        
    def scale(self,mat):
        scaled_mat = (mat.T/self.scaling_vec[:,None]).T
        return scaled_mat
         
    def logLossError(self, act, pred):
        epsilon = 1e-15
        pred = sp.maximum(epsilon, pred)
        pred = sp.minimum(1-epsilon, pred)
        ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
        ll = ll * -1.0/len(act)
        return ll
        
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
        
    def learn(self,no_iter,shuffle, eta=0.001, theta=0.00001):
        '''
        note: logistic loss function is used by default
        no_iter : how many times the algorithm is applied to the training set
        shuffle : boolean indicating if training shall be shuffled before each iteration
        '''            
        #eta   = 0.001  # learning rate
        #theta = 0.7    # regularization parameter
        new_error=1
        for it in range(0,no_iter):
            res=self.feature_mat.dot(self.weights_vec)
            old_error = new_error
            new_error=self.logLossError(self.label_vec,res)
            delta_error = np.fabs(old_error - new_error)
            print('iteration : '+str(it))
            print('Error with latest weights:'+str(new_error))
            print('Error delta with last error:'+str(delta_error))

            #convergence criteria            
            if delta_error < 0.01*eta:
                break
            
            if shuffle == True:
                combi_mat =  np.append(self.label_vec[:,None],self.feature_mat,1)
                np.random.shuffle( combi_mat )
                self.feature_mat = combi_mat[:,1:] 
                self.label_vec   = combi_mat[:,0]
            # run SGD for every sample
            for ind, row in enumerate(self.feature_mat):
                p = np.inner(self.weights_vec,row) # compute probability
                alpha = eta / ( float(self.label_vec.shape[0]) * (1.0+eta*theta*self.iter) )
                self.weights_vec -= alpha * theta*self.weights_vec #l2 Regularization
                self.weights_vec -= alpha * self.logistic_loss_function_gradient(row,self.label_vec[ind],p)
            
            self.iter += 1
        return self.weights_vec
        
        
    def predict(self,test_id,test_features):
    
        if np.linalg.norm(self.weights_vec) == 0:
            raise ValueError('All weights are zero! (Possible reason: learn() was not called.)')
        # scale feature mat by scaling_vec (the same as used for the training set)
        test_features = self.scale(test_features)
        return ( np.matrix([ test_id.astype(int), np.dot(test_features,self.weights_vec) ])).T
        
        
     
        
    
