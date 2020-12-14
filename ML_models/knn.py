#----------------------Imports--------------------------

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Uncomment the line below if running as a jupyter notebook
# %matplotlib inline 

#----------------------Read Data------------------------

def read_dataset(feature_file, label_file):
    '''Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values 
    y = df_y.values 
    return X, y


def normalize_features(X_train, X_test):
    '''Normalize the data'''
    from sklearn.preprocessing import StandardScaler 
    scaler = StandardScaler() 
    scaler.fit(X_train) 
    X_train_norm = scaler.transform(X_train) 
    X_test_norm = scaler.transform(X_test) 
    return X_train_norm, X_test_norm

# Choose your dataset here

# X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
# X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)

X_train_norm, X_test_norm = normalize_features(X_train, X_test)

#------------------------Classes--------------------------

class kNN:
    def __init__(self, X_train, y_train, X_test, k, mode):
        '''
        k = number of neighbors
        mode = which distance metric is used
        mode options: 'eucl', 'manhattan'
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.k = k
        self.mode = mode
        
    def eucl_dist(self, pt1, pt2):
        '''Standard euclidean distance'''
        dist = np.sqrt(np.sum((pt1 - pt2)**2))        
        return dist
    
    def manhattan_dist(self, pt1, pt2):
        '''Taxi cab distance, in other words l1'''
        dist = np.sum(np.absolute(pt1 - pt2))
        return dist
        
    def get_neighbors_labels(self, test_pt):
        '''Finds the k nearest neighbors under the specified distance metric and returns their labels'''
        distances = np.zeros(self.X_train.shape[0])
        
        if self.mode == 'eucl':
            for i in range(self.X_train.shape[0]):
                dist_one_pt = self.eucl_dist(test_pt, self.X_train[i]) # Calculates the distance from every point
                distances[i] = dist_one_pt # Stores the distances
                
        if self.mode == 'manhattan':
            for i in range(self.X_train.shape[0]):
                dist_one_pt = self.manhattan_dist(test_pt, self.X_train[i]) # Calculates the distance from every point
                distances[i] = dist_one_pt # Stores the distances
                
        neighbors_indices = np.argsort(distances)[: self.k] # Gets the indices of the k nearest points
        neighbors_labels = np.zeros(neighbors_indices.shape) 
        
        for i in range(len(neighbors_indices)):
            neighbors_labels[i] = self.y_train[neighbors_indices[i]] # Gets the labels of sampleswith those indices
            
        return neighbors_labels
    
    def vote(self, labels):
        '''Returns the most common label from the list of k nearest neighbor labels'''
        # Counter counts how many times each item appears in a list and returns a dictionary containing this information
        counted_labels = Counter(labels) 
        # Counted_labels.most_common(n) gives a list of the n most common elements
        return counted_labels.most_common(1)[0][0] 
        
    def predict(self):
        '''Returns a prediction of the labels for the test data'''
        ypred = np.zeros(self.X_test.shape[0])
        for i in range(self.X_test.shape[0]): # Runs through each point in the test set
            labels = self.get_neighbors_labels(self.X_test[i]) # Getting the labels of nearest neighbors in the train set
            vote_winner = self.vote(labels)
            ypred[i] = vote_winner # The winner of the vote is the prediction!
        return ypred
    
#----------------------Functions------------------------

def accuracy(ypred, yexact):
    '''
    Returns the accuracy of our model 
    ypred = labels obtained from feeding the test data forward with learned weights
    yexact = true labels from test set
    '''
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))    

#----------------------Training-------------------------
    
my_kNN = kNN(X_train_norm, y_train, X_test_norm, k=3, mode='manhattan')

ypred = my_kNN.predict()


print('Accuracy of our model ', accuracy(ypred, y_test.ravel()))

