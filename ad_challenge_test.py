import numpy as np
import scipy.stats as stats


import algorithms # our own implementation of the algorithms

path = '/Users/Sweta/Documents/Kaggle/Advertising_Challenge/'

def convertfunc(value):
    if value != np.nan:
        return float(int(value,16))

# fill 'nan': use the mean of the column
def fillna_with_mean(mat):   
    #Obtain mean of columns as you need, nanmean is just convenient.
    col_mean = stats.nanmean(mat,axis=0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(mat))
    #Place column means in the indices. Align the arrays using take
    mat[inds]=np.take(col_mean,inds[1])

'''
Read training data
'''
conv = {col: convertfunc for col in range(1,41)}
train_data = np.genfromtxt(path+'train_head.csv',delimiter=',',skip_header=1,dtype=float,converters=conv)
# fill 'Nan' values with the mean of the corresponding column
fillna_with_mean(train_data[:,2:])
# define training labels and features
train_labels   = train_data[:,1]
train_features = train_data[:,2:]

'''
Read test data
'''
conv = {col: convertfunc for col in range(1,40)}
test_data = np.genfromtxt(path+'test_head.csv',delimiter=',',skip_header=1,dtype=float,converters=conv)   
# fill 'Nan' values with the mean of the corresponding column  
fillna_with_mean(test_data[:,1:])
# define test id and features
test_id       = test_data[:,0]
test_features = test_data[:,1:]


'''
Learn and test part
'''

# initialize stochastic gradient descent class (home made)
sgd = algorithms.StochasticGradientDescentMethod(train_labels,train_features)
sgd.learn(10,True)

result = sgd.predict(test_id,test_features)
print result[:20,:]

































