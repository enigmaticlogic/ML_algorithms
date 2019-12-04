import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor as RF
from scipy import stats

# 2007 dataset
# X_train_file = '/mnt/home/storeyd3/Documents/FRI_Project/v2007_features/X_train'
# y_train_file = '/mnt/home/storeyd3/Documents/FRI_Project/v2007_features/y_train'
# X_test_file = '/mnt/home/storeyd3/Documents/FRI_Project/v2007_features/X_test'
# y_test_file = '/mnt/home/storeyd3/Documents/FRI_Project/v2007_features/y_test'

# 2013 dataset
X_train_file = '/mnt/home/storeyd3/Documents/FRI_Project/v2013_features/X_train'
y_train_file = '/mnt/home/storeyd3/Documents/FRI_Project/v2013_features/y_train'
X_test_file = '/mnt/home/storeyd3/Documents/FRI_Project/v2013_features/X_test'
y_test_file = '/mnt/home/storeyd3/Documents/FRI_Project/v2013_features/y_test'

with open(X_train_file, 'rb') as handle:
	X_train = pickle.load(handle)
with open(y_train_file, 'rb') as handle:
	y_train = pickle.load(handle)
with open(X_test_file, 'rb') as handle:
	X_test = pickle.load(handle)
with open(y_test_file, 'rb') as handle:
	y_test = pickle.load(handle)

pCC_array = np.zeros(50)
for index in range(50):
	RF_reg = RF(n_estimators = 550).fit(X_train, y_train)
	RF_ypred = RF_reg.predict(X_test)
	RF_pCC = stats.pearsonr(RF_ypred, y_test)[0]
	print(100*index/50, 'percent done')
	pCC_array[index] = RF_pCC

avg_pCC = np.average(pCC_array)
print('The avg pCC for the 2013 dataset is', avg_pCC)