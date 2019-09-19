import numpy as np
import pandas as pd
%matplotlib inline 

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

def one_hot_encoder(y_train, y_test):
    ''' convert label to a vector under one-hot-code fashion '''
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    y_train_ohe = lb.transform(y_train)
    y_test_ohe = lb.transform(y_test)
    return y_train_ohe, y_test_ohe

def accuracy(ypred, yexact):
    '''
    Returns the accuracy of our model 
    ypred = labels obtained from feeding the test data forward with learned weights
    yexact = true labels from test set
    '''
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))

X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')

X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

new_train_col = np.ones((X_train.shape[0],1)) 
new_test_col = np.ones((X_test.shape[0],1)) 
X_train_norm = np.append(X_train_norm, new_train_col, 1)
X_test_norm = np.append(X_test_norm, new_test_col, 1)

class Log_Reg:
    def __init__(self, X, y, c, lr=0.01, reg=0.1):
        self.X = X
        self.y = y
        self.c = c
        self.lr = lr
        self.reg = reg
        
        self.num_data_pts = self.X.shape[0]
        self.loss_all_cols = np.zeros(10)
        
    def sigmoid(self, z):
        '''Sigmoid predictor function'''
        return 1 / (1 + np.exp(-z))
        
    def binary_gd_fit(self, y_one_col, c_one_col):
        '''Does gradient descent for one column'''
        z = np.dot(self.X, c_one_col)
        grad = np.dot(self.X.T, (self.sigmoid(z) - y_one_col))/self.num_data_pts
        return grad
        
    def loss(self, y_one_col, c_one_col):
        z = np.dot(self.X, c_one_col)
        current_loss = (-np.dot(y_one_col, np.log(0.0001 + self.sigmoid(z))) - np.dot((1 - y_one_col),\
        np.log(1.0001 - self.sigmoid(z))))/(self.num_data_pts)
        return current_loss
                                   
    def log_reg_OVR_train(self):
        '''Runs each column through the binary gradient descent'''
        cgrad = self.c
        for i in range(self.y.shape[1]): # Running through ten columns
            y_train_one_col = self.y[:,i] 
            c_input_col = self.c[:,i]
            cgrad_one_col = self.binary_gd_fit(y_train_one_col, c_input_col)
            cgrad[:,i] = cgrad_one_col # Stores the gradient from one column
            loss_one_col = self.loss(y_train_one_col, c_input_col)
            self.loss_all_cols[i] = loss_one_col
        self.c = self.c - self.lr*cgrad # Updating the wieghts of all the columns    
        print('Current loss: ', np.sum(self.loss_all_cols))
                                     
    def predict(self, X_test):
        '''Uses the learned weights to do a one vs all prediction'''
        num_test_samples = X_test.shape[0]
        labels = np.zeros((num_test_samples, self.y.shape[1]))
        for i in range(self.y.shape[1]): # Running through ten columns
            z = np.dot(X_test, self.c[:,i])
            labels[:,i]  = self.sigmoid(z) # Putting the probabilities into labels
        ypred = np.zeros(num_test_samples, dtype=int) 
        for i in range(num_test_samples): # Running through the data points
            ypred[i] = np.argmin(labels[i,:]) # 
        return ypred

def accuracy(ypred, yexact):
    p = np.array(ypred == yexact, dtype = int)
    return np.sum(p)/float(len(yexact))
        
c = np.random.randn(X_train_norm.shape[1], 10)

my_Log_Reg = Log_Reg(X_train_norm, y_train_ohe, c, lr=0.01, reg=0.5)

for epoch in range(1000):
    print("Epoch number ", epoch+1)
    my_Log_Reg.log_reg_OVR_train()


ypred = my_Log_Reg.predict(X_test_norm)
print('Accuracy of our model ', accuracy(ypred, y_test.ravel()))
    