#----------------------Imports-------------------------

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import datasets
# Uncomment the line below if running as a jupyter notebook
# %matplotlib inline 

#----------------------Read_Data-----------------------

def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values # convert values in dataframe to numpy array (features)
    y = df_y.values # convert values in dataframe to numpy array (label)
    return X, y

def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler #import libaray
    scaler = StandardScaler() # call an object function
    scaler.fit(X_train) # calculate mean, std in X_train
    X_train_norm = scaler.transform(X_train) # apply normalization on X_train
    X_test_norm = scaler.transform(X_test) # we use the same normalization on X_test
    return X_train_norm, X_test_norm

X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')

X_train_norm, X_test_norm = normalize_features(X_train, X_test)

#----------------------Classes-------------------------

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=1, current_depth=1, mode='classifier'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.current_depth = current_depth # Needs to be initialized to 1
        self.mode = mode # 'classifier' or 'regressor'
        self.left_tree = 'unsplit' 
        self.right_tree = 'unsplit'

    def fit(self, X, y):
        ''' 
        X, y are initially the training data and training labels
        After the first pass, fit runs on the branches of the split tree
        '''
        self.X = X
        self.y = y
        self.num_samples = X.shape[0]
        self.num_features = X.shape[1]
        
        # Splits until these conditions are met
        if self.current_depth <= self.max_depth and self.num_samples >= self.min_samples_split:
            if self.mode == 'classifier':
                self.best_feature_id, self.best_score, self.best_split_value = self.find_best_split() 
                if self.calculate_GINI(self.y) > 0: # if the node is not pure, then we split with best split value
                    self.split_trees()
                    
            if self.mode == 'regressor':
                self.best_feature_id, self.best_score, self.best_split_value = self.find_best_split() 
                self.split_trees()

    def calculate_GINI(self, y):
        '''Calculates the GINI score for a branch'''
        unique, counts = np.unique(y, return_counts=True)
        prob = counts/float(y.size)
        return 1.0 - np.sum(prob**2)

    def calculate_SSE(self, y):
        '''Calculates the SSE for a branch'''
        group_mean = np.mean(y)
        return np.sum((y - group_mean)**2) 
    
    def calculate_weighted_score(self, left_tree_y, right_tree_y):
        '''Calculates the weighted GINI or SSE score of the new branches so that the gain can be tested'''
        if self.mode == 'classifier':
            left_score = self.calculate_GINI(left_tree_y) # calculate left and right GINI's
            right_score = self.calculate_GINI(right_tree_y)
            
        if self.mode == 'regressor':
            left_score = self.calculate_SSE(left_tree_y) # calculate left and right SSE's
            right_score = self.calculate_SSE(right_tree_y)
            
        weighted_score = left_score*left_tree_y.shape[0]/float(self.num_samples)\
        + right_score*right_tree_y.shape[0]/float(self.num_samples) 
        
        return weighted_score
    
    def find_best_split_one_feature(self, feature_id):
        '''Finds the best split for a single feature based on GINI or SSE scores'''
        feature_values = self.X[:, feature_id] # Gets the values of a single feature for all samples
        unique_feature_values = np.unique(feature_values) # Removes redundant values
        best_score = float('inf')
        best_split_value = None 
        
        if len(unique_feature_values) == 1:
            return best_score, best_split_value
        
        for fea_val in unique_feature_values: # Check each unique feature to see which gives the best split
            left_indices = np.where(feature_values < fea_val)[0] # Gets the indices of samples to be put in the left branch
            right_indices = np.where(feature_values >= fea_val)[0] # Gets the indices of samples to be put in the right branch
            left_tree_y = self.y[left_indices] # Making the new branches
            right_tree_y = self.y[right_indices]
            
            # Moves to the next feature if one of the new branches would have no samples in it
            if left_tree_y.shape[0] == 0 or right_tree_y.shape[0] == 0: 
                continue
            
            current_score =  self.calculate_weighted_score(left_tree_y, right_tree_y)
            
            if best_score > current_score: # If the gain is positive, the current score becomes the new best score
                best_score = current_score
                best_split_value = fea_val
                
        return best_score, best_split_value

    def find_best_split(self):
        '''
        Checks every feature to find which feature gives the best split
        Returns the feature resulting in the best split and the best split value/score
        '''
        best_feature_id = None
        best_score = float('inf')
        best_split_value = None
        
        for feature_id in range(self.num_features): # Runs through every feature
            current_score, current_split_value = self.find_best_split_one_feature(feature_id)
            
            if best_score > current_score:
                best_score = current_score
                best_split_value = current_split_value
                best_feature_id = feature_id
                
        return best_feature_id, best_score, best_split_value 

    def split_trees(self):
        '''Creates the new branches after the best split parameters are identified'''
        # Initialize the new branches
        self.left_tree = DecisionTree(self.max_depth, self.min_samples_split, self.current_depth+1, self.mode)
        self.right_tree = DecisionTree(self.max_depth, self.min_samples_split, self.current_depth+1, self.mode)
        
        # Split the data using the best split value
        if self.best_split_value == None:
            self.left_tree = 'unsplit' 
            self.right_tree = 'unsplit'
            return None        
        
        best_feature_values = self.X[:, self.best_feature_id]
        left_indices = np.where(best_feature_values < self.best_split_value)[0]
        right_indices = np.where(best_feature_values >= self.best_split_value)[0]
        left_tree_X = self.X[left_indices]
        left_tree_y = self.y[left_indices]
        right_tree_X = self.X[right_indices]
        right_tree_y = self.y[right_indices]
        
        # Recursively fits and creates new splits
        self.left_tree.fit(left_tree_X, left_tree_y) 
        self.right_tree.fit(right_tree_X, right_tree_y)

    def predict_label(self): 
        '''Predicts the label of a leaf node'''
        if self.mode == 'classifier':
            unique, counts = np.unique(self.y, return_counts=True) # Find the labels and counts of samples in the leaf node
            label = None 
            max_count = 0
            
            for i in range(unique.size): # Runs through each unique label
                if counts[i] > max_count: # Finds the label that has the highest count
                    max_count = counts[i]
                    label = unique[i]
                    
            return label
        
        if self.mode == 'regressor':
            return np.mean(self.y)

    def tree_propagation(self, features):
        '''Propagates one piece of test data through the constructed decision tree'''
        
        # If the node is unsplit, it's a leaf and we predict
        if self.left_tree == 'unsplit': 
            return self.predict_label()
        
        # Otherwise we move to a new branch based on our best split parameters
        if features[self.best_feature_id] < self.best_split_value: 
            child_tree = self.left_tree
            
        else:
            child_tree = self.right_tree
            
        return child_tree.tree_propagation(features) # Recursively call the tree propagation until we hit a leaf node

    def predict(self, X_test):
        '''Runs each test data point through the tree to predict'''
        num_test_samples = X_test.shape[0]
        ypred = np.zeros(num_test_samples)
        for i in range(num_test_samples):
            ypred[i] = self.tree_propagation(X_test[i])
        return ypred

#----------------------Functions-----------------------

def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

#-----------------------Training-------------------------------

myTree = DecisionTree(max_depth=11, min_samples_split=1, mode='classifier')
myTree.fit(X_train_norm, y_train)
ypred = myTree.predict(X_test_norm)
print("y prediction: ", ypred)
print("accuracy: ", accuracy(ypred, y_test.ravel()))

