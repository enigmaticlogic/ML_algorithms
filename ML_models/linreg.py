import numpy as np
import pandas as pd
# Uncomment the line below if running as a jupyter notebook
# %matplotlib inline 

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

X_train, y_train = read_dataset('airfoil_self_noise_X_train.csv', 'airfoil_self_noise_y_train.csv')
X_test, y_test = read_dataset('airfoil_self_noise_X_test.csv', 'airfoil_self_noise_y_test.csv')

X_train_norm, X_test_norm = normalize_features(X_train, X_test)

new_train_col = np.ones((X_train.shape[0],1)) 
new_test_col = np.ones((X_test.shape[0],1)) 
X_train_norm = np.append(X_train_norm, new_train_col, 1)
X_test_norm = np.append(X_test_norm, new_test_col, 1)

class Lin_Reg:
    def __init__(self, X, y, c, lr=0.01, mode='OLR', reg=0.1):
        self.X = X
        self.y = y
        self.c = c
        self.lr = lr
        self.mode = mode 
        self.reg = reg
        
        self.num_data_pts = self.X.shape[0]
    
    def gd_fit(self):
        '''Ordinary linear regression'''
        intermediate = np.dot(self.X, self.c) - self.y 
        self.grad = np.dot(self.X.T, intermediate)/self.num_data_pts # Vectorized gradient
        self.c = self.c - self.lr*self.grad
        
    def ridge_gd_fit(self):
        '''Linear regression with L2 regularization'''
        intermediate = np.dot(self.X, self.c) - self.y 
        self.grad = np.dot(self.X.T, intermediate)/self.num_data_pts + self.reg*self.c/self.num_data_pts
        self.c = self.c - self.lr*self.grad
        
    def loss(self):
        '''Loss without regularization'''
        self.current_loss = np.sum((self.X.dot(self.c) - self.y) ** 2)/(2 * self.num_data_pts)
        
    def ridge_loss(self):
        '''Loss with regularization'''
        self.ridge_current_loss = np.sum((self.X.dot(self.c) - self.y) ** 2)/(2 * self.num_data_pts) \
        + (self.reg * np.dot(self.c.T, self.c)[0][0])/(2 * self.num_data_pts) 
        
           
    def predict(self, X_test):
        '''Uses learned weights to predict'''
        prediction = np.dot(X_test, self.c)
        return prediction
        
def RMSE(y_pred, y_test):
    '''Finds the root mean squared error as a measure of accuracy'''
    diff = y_pred - y_test
    return np.sqrt(sum(diff*diff)/y_pred.shape[0])
        
c = np.random.randn(X_train_norm.shape[1], 1) # Initialize the weights

my_Lin_Reg = Lin_Reg(X_train_norm, y_train, c, lr=0.01, mode='Ridge', reg=0.5) # Adjust parameters here

for epoch in range(1000):
    print("Epoch number ", epoch+1)
           
    if my_Lin_Reg.mode =='OLR':
        my_Lin_Reg.gd_fit()
        my_Lin_Reg.loss()
        print("loss: ", my_Lin_Reg.current_loss)
           
    if my_Lin_Reg.mode =='Ridge':
        my_Lin_Reg.ridge_gd_fit()
        my_Lin_Reg.ridge_loss()
        print("loss: ", my_Lin_Reg.ridge_current_loss)

ypred = my_Lin_Reg.predict(X_test_norm)

print("RMSE: ", RMSE(ypred, y_test)[0])

    
