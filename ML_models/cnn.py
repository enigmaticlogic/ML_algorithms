import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt 
from skimage.measure import block_reduce
from random import shuffle
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
    '''Convert label to a vector under one-hot-code fashion'''
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

'''Load the data sets and normalize them'''
# X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
# X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')

X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')

X_train_norm, X_test_norm = normalize_features(X_train, X_test)
y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

class conv_NN:
    def __init__(self, X, y, hidden_nn_1=100, hidden_nn_2=100, conv_h=3, conv_w=3,\
                 pool_h=2, pool_w=2, num_filters=1, lr=0.01, reg=.1):
        self.X = X 
        self.y = y 
        self.hidden_nn_1 = hidden_nn_1
        self.hidden_nn_2 = hidden_nn_2
        self.conv_h = conv_h
        self.conv_w = conv_w
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.num_filters = num_filters
        self.lr = lr
        self.reg = reg
        
        self.num_images = X.shape[0]
        self.input_nn = X.shape[1] # Number of neurons in the input layer
        self.output_nn = y.shape[1] # Number of neurons in the output layer
        self.X_reshaped = X.reshape(self.num_images, 1, int(np.sqrt(self.input_nn)), -1) # Reshaping input for convolution
        
        '''
        Initialize all the weight arrays to be the correct size with random values,
        scaled down according to the number of inputs
        '''
        # For the convolution layer
        self.C1 = np.random.randn(self.num_filters, self.conv_h, self.conv_w)/np.sqrt(self.conv_h)
        self.Cb1 = np.zeros((self.num_filters, 1))
        
        # For the ANN layers
        self.W1 = np.random.randn((self.num_filters*self.X_reshaped[0, :, :, :].size)\
        //(self.pool_h*self.pool_w), hidden_nn_1)/np.sqrt((num_filters*self.X_reshaped[0, :, :, :].size)//(pool_h*pool_w))
        self.b1 = np.zeros((1, self.hidden_nn_1))
        self.W2 = np.random.randn(self.hidden_nn_1, self.hidden_nn_2) / np.sqrt(self.hidden_nn_1)
        self.b2 = np.zeros((1, self.hidden_nn_2))
        self.W3 = np.random.randn(self.hidden_nn_2, self.output_nn) / np.sqrt(self.hidden_nn_2)
        self.b3 = np.zeros((1, self.output_nn))
        
    def add_padding(self, X): 
        '''Adds a pad of zeros to the image so it can be convolved properly'''
        # Only pads the last two indices, since these are the image pixels
        X = np.pad(X, ((0,0),(0,0),(1,1),(1,1)), 'constant', constant_values=0) 
        return X
    
    def max_pooling(self, X): 
        '''Returns an array reduced via max pooling'''
        return block_reduce(X, (1, 1, self.pool_h, self.pool_w), np.max)

    def back_prop_pooling(self, X, err_0): 
        '''
        Returns to the original size of the image from before max pooling was performed
        Credit to Timothy Szocinski for this function
        '''
        for s in range(X.shape[0]): # Loop through each image
            for t in range(self.num_filters): # Loop through filters
                for i in range(X.shape[2]//2): # Loop through each 2x2 subarray
                    for j in range(X.shape[3]//2): 
                        # Applies argmax to each 2x2 block and records the index of the max value
                        ind = np.unravel_index(np.argmax(X[s, t, 2*i:2*i+2, 2*j:2*j+2]), X[s, t, 2*i:2*i+2, 2*j:2*j+2].shape)
                        # Creates an array of zeros at all indices aside from where the max values were
                        X[s, t, 2*i:2*i+2, 2*j:2*j+2] = 0 
                        X[s, t, 2*i+ind[0], 2*j+ind[1]] = err_0[s, t, i, j] 
        return X
    
    def convolve_ff(self, X):
        '''Cycles through each image and filter and convolves them in the feed forward'''
        conv = np.zeros((X.shape[0], self.num_filters, X.shape[2], X.shape[3])) 
        for i in range(X.shape[0]): # Loops through each image
            for j in range(self.num_filters): # Loops through each filter
                # Same mode is used to add padding to keep the dimensions of the input and output the same
                conv[i, j, :, :] = sp.signal.convolve2d(X[i, 0, :, :], self.C1[j, :, :], 'same') + self.Cb1[j, 0] 
        return conv
    
    def convolve_backprop(self, X, err_0):
        '''Cycles through each image and filter and convolves with the corresponding error term in the backprop'''
        conv = np.zeros((X.shape[0], self.num_filters, self.conv_h, self.conv_w)) 
        for i in range(X.shape[0]): # Loops through each image
            for j in range(self.num_filters): # Loops through the error for each filter
                # Input will be padded so that the output is the correct size
                conv[i, j, :, :] = sp.signal.convolve2d(X[i, 0, :, :], err_0[i, j, :, :], 'valid')
        return conv
    
    def Activator(self, z):
        '''
        The activator function can be modified
        Current activator = Relu
        '''
        z[z<0] = 0
        return 0.1*z # Output can be scaled here
    
    def Activator_prime(self, z):
        '''
        Derivative of the activator function
        Current activator = Relu
        '''
        z[z<0] = 0
        z[z>0] = 1
        return 0.1*z
    
    def softmax(self, z):
        '''Activator for the output layer'''
        exp_value = np.exp(z-np.amax(z, axis=1, keepdims=True))
        softmax_scores = exp_value / np.sum(exp_value, axis=1, keepdims=True)
        return softmax_scores
    
    def feed_forward(self):
        '''
        Feeds the data through the network, starting with the 
        reshaped image data and ending with a one hot vector
        '''
        # Convolutional layer part
        self.z0 = self.convolve_ff(self.X_reshaped) 
        self.f0_preshaped = self.Activator(self.z0)
        self.f0 = self.max_pooling(self.f0_preshaped) 
        self.f0 = self.f0.reshape(self.f0.shape[0], -1) 
        
        # ANN part
        self.z1 = np.dot(self.f0, self.W1) + self.b1
        self.f1 = self.Activator(self.z1)
        self.z2 = np.dot(self.f1, self.W2) + self.b2    
        self.f2 = self.Activator(self.z2)
        self.z3 = np.dot(self.f2, self.W3) + self.b3
        self.y_hat = self.softmax(self.z3)
        
    def back_propagation(self):
        '''Back propagates the error from each layer to calculate gradients for gradient descent'''
        # Derivatives of the activated terms from the feed forward
        dz0 = self.Activator_prime(self.z0)
        dz1 = self.Activator_prime(self.z1)
        dz2 = self.Activator_prime(self.z2)
        
        # Calculating the error at each layer
        err_3 = self.y_hat - self.y
        err_2 = np.dot(err_3, (self.W3).T)*dz2
        err_1 = np.dot(err_2, (self.W2).T)*dz1
        err_0 = np.dot(err_1, (self.W1).T)
        err_0 = err_0.reshape(err_0.shape[0], self.num_filters, int(np.sqrt(err_0.shape[1]/self.num_filters)), -1)
        err_0 = self.back_prop_pooling(self.f0_preshaped, err_0)
        
        # Calculating the gradient for each weight
        dW3 = np.dot((self.f2).T, err_3) + self.reg*self.W3/self.X.shape[0]
        dW2 = np.dot((self.f1).T, err_2) + self.reg*self.W2/self.X.shape[0]
        dW1 = np.dot((self.f0).T, err_1) + self.reg*self.W1/self.X.shape[0]
        # Take the sum to total the gradient contributions from each image and filter since we have shared weights
        dC1 = np.sum(self.convolve_backprop(self.add_padding(self.X_reshaped), err_0*dz0), axis=0) 
        
        # Bias gradients
        db3 = np.sum(err_3, axis=0, keepdims=True)
        db2 = np.sum(err_2, axis=0, keepdims=True)
        db1 = np.sum(err_1, axis=0, keepdims=True)
        dCb1 = np.sum(err_0*dz0)
        
        # Updating the weights and biases!
        self.C1 = self.C1 - self.lr * dC1
        self.Cb1 = self.Cb1 - self.lr * dCb1
        self.W1 = self.W1 - self.lr * dW1
        self.b1 = self.b1 - self.lr * db1
        self.W2 = self.W2 - self.lr * dW2
        self.b2 = self.b2 - self.lr * db2
        self.W3 = self.W3 - self.lr * dW3
        self.b3 = self.b3 - self.lr * db3
    
    def convolution_fit(self):
        '''Fit function which uses feed forward and back prop to update the weights and biases a single time'''
        self.feed_forward()
        self.back_propagation()
                        
    def convolution_minibatch_fit(self, batch_size):
        '''Uses mini batches to feed a small number of randomly selected data points through the network'''
        self.feed_forward() 
        y_hat = self.y_hat 
        y = self.y 
        X_reshaped = self.X_reshaped
        order = np.arange(self.num_images) # Gather indices into range array
        shuffle(order) # Shuffle indices for random batch
        self.y = self.y[order, :] # Randomizes the order of the labels by using shuffled index
        self.y = self.y[0:batch_size, :] # Takes a minibatch of labels of size batch_size
        self.X_reshaped = self.X_reshaped[order, :, :, :] # Does the same for the images
        self.X_reshaped = self.X_reshaped[0:batch_size, :, :, :] 
        self.convolution_fit() # Runs the mini batch through the network and updates weights
        self.y = y # Reassigns parameters to their original unshuffled state
        self.X_reshaped = X_reshaped
        self.y_hat = y_hat
        
    def convolution_predict(self, X_test):
        '''Feeds the test data through the network using learned weights and returns the predicted results'''
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        num_test_samples = X_test.shape[0]
        ypred = np.zeros(num_test_samples, dtype=int)
        
        # Concolution layer part
        X_test_reshaped = X_test.reshape(num_test_samples, 1, int(np.sqrt(X_test.shape[1])), -1)
        z0 = self.convolve_ff(X_test_reshaped)
        f0_preshaped = self.Activator(z0)
        f0 = self.max_pooling(f0_preshaped) 
        f0 = f0.reshape(num_test_samples, -1) 
        
        # ANN part
        z1 = np.dot(f0, self.W1) + self.b1
        f1 = self.Activator(z1)
        z2 = np.dot(f1, self.W2) + self.b2    
        f2 = self.Activator(z2)
        z3 = np.dot(f2, self.W3) + self.b3
        y_hat_test = self.softmax(z3)
    
        for i in range(num_test_samples):
            ypred[i] = labels[np.argmax(y_hat_test[i, :])]
        return ypred
    
    
    def cross_entropy_loss(self):
        '''Calculates the loss given the current predictions'''
        self.loss = -np.sum(self.y*np.log(self.y_hat+1e-6))

'''
Initialize parameters here
hidden_nn_1 = number of neurons in first layer of the ANN
hidden_nn_2 = number of neurons in the second layer of the ANN
conv_h = height of the convolution filters
conv_w = width of the convolution filters
pool_h = height of the pooling filter
pool_w = width of the pooling filter
num_filters = number of convolution filters to be applied in the convolution layer
lr = learning rate, determines how much the weights are updated at each gradient descent step
reg = regularization constant, used to combat overfitting
'''
my_NN = conv_NN(X_train_norm, y_train_ohe, hidden_nn_1=100, hidden_nn_2=100, conv_h=3, conv_w=3,\
                pool_h=2, pool_w=2, num_filters=16, lr=0.1, reg=.000001)

'''Choose how many times to update the weights and the minibatch size here'''
for epoch in range(3000):
    my_NN.convolution_minibatch_fit(20)
    my_NN.cross_entropy_loss()
    if (epoch%50) == 0:
        print("Epoch number ", epoch+1)
        print("loss: ", my_NN.loss)
    
ypred = my_NN.convolution_predict(X_test_norm)
print(ypred)
print(accuracy(ypred, y_test.ravel()))
