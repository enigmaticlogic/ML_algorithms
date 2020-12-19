import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
# Uncomment the line below if running as a jupyter notebook
# %matplotlib inline 

def read_dataset(feature_file, label_file):
    ''' Read data set in *.csv to data frame in Pandas'''
    df_X = pd.read_csv(feature_file)
    df_y = pd.read_csv(label_file)
    X = df_X.values 
    y = df_y.values 
    return X, y


def normalize_features(X_train, X_test):
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


# X_train, y_train = read_dataset('MNIST_X_train.csv', 'MNIST_y_train.csv')
# X_test, y_test = read_dataset('MNIST_X_test.csv', 'MNIST_y_test.csv')

X_train, y_train = read_dataset('Digits_X_train.csv', 'Digits_y_train.csv')
X_test, y_test = read_dataset('Digits_X_test.csv', 'Digits_y_test.csv')

X_train_norm, X_test_norm = normalize_features(X_train, X_test)

X_train_norm = torch.tensor(X_train_norm, dtype=torch.float32)
X_test_norm = torch.tensor(X_test_norm, dtype=torch.float32)

y_train_ohe, y_test_ohe = one_hot_encoder(y_train, y_test)

y_train_ohe = torch.tensor(y_train_ohe, dtype=torch.float32)
y_test_ohe = torch.tensor(y_test_ohe, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int)


class layers:
    def __init__(self, W, b, f_in=0, err=0, err_out=0, W_out=0, hidden_layer_nn=100, reg=.05, lr=.01):
        self.W = W
        self.b = b
        self.f_in = f_in
        self.err = err
        self.err_out = err_out
        self.W_out = W_out
        self.hidden_layer_nn = hidden_layer_nn
        self.reg = reg
        self.lr = lr
        
        
    def activator(self, z):
        z = z.numpy()
        z = z*(z>0)
        z = torch.from_numpy(z)
        return z
    
    def activator_deriv(self, z):
        z = z.numpy()
        z[z<0] = 0
        z[z>0] = 1
        z = torch.from_numpy(z)
        return z
        
    def feed_forward(self):
        self.z = torch.mm(self.f_in, self.W) + self.b
        self.f = layers.activator(self, self.z)
        
    def back_propagation(self):
        dz = layers.activator_deriv(self, self.z)
        self.err = torch.mm(self.err_out, (self.W_out).transpose(0,1))*dz 
        self.dW = torch.mm((self.f_in).transpose(0,1), self.err) + self.reg*self.W/self.hidden_layer_nn
        self.db = torch.sum(self.err, dim=0, keepdim=True)
    
    def updater(self):
        self.W = self.W - self.lr * self.dW
        self.b = self.b - self.lr * self.db

class output_layer(layers):
    def __init__(self, W, b, f_in=0, err=0, err_out=0, W_out=0, hidden_layer_nn=100, reg=.05, lr=.01):
        super(output_layer, self).__init__(W, b, f_in=0, err=0,err_out=0, W_out=0, hidden_layer_nn=100, reg=.05, lr=.01)
        
        
    def feed_forward(self):
        self.z = torch.mm(self.f_in, self.W) + self.b
        self.f = softmax(self.z)
#         self.f = F.softmax(self.z, 1)
        
    def back_propagation(self):
        self.dW = torch.mm((self.f_in).transpose(0,1), self.err) + self.reg*self.W/self.hidden_layer_nn
        self.db = torch.sum(self.err, dim=0, keepdim=True)
    
class whole_network:
    def __init__(self, num_hidden_layers, num_hidden_neurons, epoch_num, X, y, reg=.1, lr=.01):
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_neurons = num_hidden_neurons
        self.epoch_num = epoch_num
        self.X = X
        self.y = y
        self.reg = reg
        self.lr = lr
        self.avg_hidden_neurons = sum(self.num_hidden_neurons)/self.num_hidden_layers
        self.nn = X_train.shape[1]
        self.output_layer_nn = y.shape[1]
        
        W_hidden = []
        b_hidden = []

        for i in range(self.num_hidden_layers - 1):
            W_hidden.append(torch.randn((self.num_hidden_neurons[i], self.num_hidden_neurons[i+1])) / np.sqrt(self.nn))
            b_hidden.append(torch.zeros((1, self.num_hidden_neurons[i+1])))

        W_input = torch.randn((self.nn, self.num_hidden_neurons[0])) / np.sqrt(self.nn)
        b_input = torch.zeros((1, self.num_hidden_neurons[0]))
        W_output = torch.randn((self.num_hidden_neurons[self.num_hidden_layers - 1], self.output_layer_nn)) \
        / np.sqrt(self.num_hidden_neurons[self.num_hidden_layers - 1])
        b_output = torch.zeros((1, self.output_layer_nn))

        self.layer_list = [layers(W_hidden[i], b_hidden[i], f_in=0, err=0, err_out=0, W_out=0, \
                             hidden_layer_nn=self.avg_hidden_neurons, reg=self.reg, lr=self.lr) \
                      for i in range(num_hidden_layers-1)]

        self.layer_list.insert(0, layers(W_input, b_input))
        self.layer_list.append(output_layer(W_output, b_output))
        
    def whole_network_feed_forward(self, X):
        self.layer_list[0].f_in = X
        for i in range(self.num_hidden_layers + 1):
            self.layer_list[i].feed_forward()
            if i != self.num_hidden_layers:
                self.layer_list[i+1].f_in = self.layer_list[i].f
        return self.layer_list[self.num_hidden_layers].f


    def whole_network_back_propagation(self, y, y_hat,):
        self.layer_list[self.num_hidden_layers].err = (y_hat - y)/self.nn
        for i in range(self.num_hidden_layers + 1):
            self.layer_list[self.num_hidden_layers - i].back_propagation()      
            if i != self.num_hidden_layers:
                self.layer_list[self.num_hidden_layers - i - 1].err_out = self.layer_list[self.num_hidden_layers - i].err
                self.layer_list[self.num_hidden_layers - i - 1].W_out = self.layer_list[self.num_hidden_layers - i].W
        for i in range(self.num_hidden_layers + 1):
            self.layer_list[self.num_hidden_layers - i].updater()
            
    def fit(self):
        '''
        X = X_train_norm
        y = y_train_ohe
        '''
        for i in range(self.epoch_num):
            y_hat = self.whole_network_feed_forward(self.X)
            self.whole_network_back_propagation(self.y, y_hat)
            loss = cross_entropy_loss(self.y, y_hat, self.layer_list, self.avg_hidden_neurons)
            if ((i+1)%50 == 0):
                print('epoch = %d, current loss = %.5f' % (i+1, loss))
                
    def prediction(self, X):
        '''
        X = X_test_norm
        '''
        y_hat_test = self.whole_network_feed_forward(X)
        return y_hat_test
       
def softmax(z):
    z = z.numpy()
    exp_value = np.exp(z-np.amax(z, axis=1, keepdims=True))
    softmax_scores = exp_value / np.sum(exp_value, axis=1, keepdims=True)
    softmax_scores = torch.from_numpy(softmax_scores)
    return softmax_scores

def cross_entropy_loss(y, y_hat, layer_list, avg_hidden_neurons):
    reg_loss = 0
    for i in range(len(layer_list)):
        reg_loss += torch.sum(torch.pow(layer_list[i].W, 2))
    loss = -torch.sum(y*torch.log(y_hat + 1e-6))/X_train_norm.shape[1] + layer_list[1].reg*(0.5)*reg_loss/avg_hidden_neurons    
    return loss

def predict(X_test, y_hat_test):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_test_samples = X_test.shape[0]
    ypred = torch.zeros(num_test_samples, dtype=torch.int) 
    for i in range(num_test_samples):
        ypred[i] = labels[torch.argmax(y_hat_test[i,:])]
    return ypred

def accuracy(ypred, yexact):
    p = torch.tensor(ypred == yexact, dtype = torch.int)
    summed = torch.sum(p)
    return float(torch.sum(p))/float(len(yexact))

myNN = whole_network(2, [100, 100], 2500, X_train_norm, y_train_ohe, reg=0.1, lr=0.01)
myNN.fit()
y_hat_test = myNN.prediction(X_test_norm)
y_pred = predict(X_test_norm, y_hat_test)

print('Accuracy of our model ', accuracy(y_pred, y_test.view(-1)))
