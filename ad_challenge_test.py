import numpy as np
import scipy.stats as stats


import algorithms # our own implementation of the algorithms

trainpath = '/Volumes/INTENSO/'
#trainpath = './'
filename = trainpath+'train.csv'

testpath = '/Volumes/INTENSO/'
#testpath = './'
testfilename = testpath+'test.csv'

resultfilename = testpath+'result.csv'

batch_length = 200000

# fill 'nan': use the mean of the column
def fillna_with_mean(mat):   
    #Obtain mean of columns as you need, nanmean is just convenient.
    col_mean = stats.nanmean(mat,axis=0)
    #Find indicies that you need to replace
    inds = np.where(np.isnan(mat))
    #Place column means in the indices. Align the arrays using take
    mat[inds]=np.take(col_mean,inds[1])

# determine the length of the file
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
            
#filter to just use a portion of the file for calculations
def filter_training_lines(f, jump):
    for i, line in enumerate(f):
        if i > 0 and i % jump == 0:
            yield line
            
#filter to just use a portion of the file for calculations
def filter_prediction_lines(f, start_val, end_val):
    for i, line in enumerate(f):
        if i >= start_val and i <= end_val:
            yield line
        elif i > end_val:
            break

# float conversion function to be used while reading in the file
def convertfunc(value):
    if value != np.nan:
        return float(int(value,16))

#columnd definition for conversion function usage
conv = {col: convertfunc for col in range(1,41)}

#determine the training set file length
file_length = file_len(filename)
file_length = batch_length + 1
print('Length of training set: '+str(file_length))

# initialize stochastic gradient descent class (home made)
sgd = algorithms.StochasticGradientDescentMethod()
#sgd.setTrainSize(file_length)
'''
Train with stochastic gradient descent
'''   
with open(filename) as f:
    train_data = np.genfromtxt(filter_training_lines(f, 100),delimiter=',',skip_header=0,dtype=float,converters=conv)

# fill 'Nan' values with the mean of the corresponding column
fillna_with_mean(train_data[:,2:])
# define training labels and features
train_labels   = train_data[:,1]
train_features = train_data[:,2:]

sgd.setTrainData(train_labels,train_features,True)
sgd.learn(100,True,eta=0.01,theta=0.000001)
print ('SGD Learning Ready')


'''
Calculate Prediction with test Data
'''

#determine the training set file length
test_file_length = file_len(testfilename)

conv = {col: convertfunc for col in range(1,40)}

start=1
end=0

with open(resultfilename, "w") as text_file:
        text_file.write('Id,Predicted\n')

while end < test_file_length:
    end += batch_length
    with open(testfilename) as f:
        test_data = np.genfromtxt(filter_prediction_lines(f, start, end),delimiter=',',skip_header=0,dtype=float,converters=conv)   
    
    # fill 'Nan' values with the mean of the corresponding column  
    fillna_with_mean(test_data[:,1:])
    
    # define test id and features
    test_id = test_data[:,0]
    test_id = test_id.astype(int)
    test_features = test_data[:,1:]

    #Calculate the prediction    
    result = sgd.predict(test_id,test_features)
    res = np.rec.fromarrays((result[:,0].astype(int), result[:,1].astype(float)))

    with open(resultfilename, 'a') as f_handle:      
        for item in res:
            f_handle.write(str(item[0][0])+','+str(item[0][1])+'\n')  
    
    print ('Start: '+str(start)+'\nEnd: '+str(end))
    start += batch_length

































