import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor as GBT
from sklearn.metrics import mean_squared_error as MSE
from scipy import stats

# 2007 dataset double lorentz kernel
data_file = '/mnt/home/storeyd3/Documents/DG_Project/features/v2016_features/LL_v1/data_dict'

# 2007 dataset double gaussian kernel
# data_file = '/mnt/home/storeyd3/Documents/DG_Project/features/v2007_features/GG_v1/data_dict'

with open(data_file, 'rb') as handle:
	data_dict = pickle.load(handle)

X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []
for dict in data_dict:
	X_train_list.append(dict['X_train'])
	y_train_list.append(dict['y_train'])
	X_test_list.append(dict['X_test'])
	y_test_list.append(dict['y_test'])

X_train = np.concatenate(X_train_list, axis=0)
print(X_train.shape)
X_test = np.concatenate(X_test_list, axis=0)
print(X_test.shape)
y_train = np.concatenate(y_train_list, axis=0)
print(y_train.shape)
y_test = np.concatenate(y_test_list, axis=0)
print(y_test.shape)

pCC_array = np.zeros(1)
RMSE_array = np.zeros(1)

for index in range(1):
	GBT_reg = GBT(loss='ls', n_estimators=10000, learning_rate=0.01, max_depth=7, min_samples_split=3, \
	  subsample=0.3, max_features='sqrt', verbose=1).fit(X_train, y_train)
	GBT_ypred = GBT_reg.predict(X_test)
	GBT_pCC = stats.pearsonr(GBT_ypred, y_test)[0]
	GBT_RMSE = np.average(np.sqrt(MSE(GBT_ypred, y_test)))
	print(100*index/50, 'percent done')
	pCC_array[index] = GBT_pCC
	RMSE_array[index] = GBT_RMSE


avg_pCC = np.average(pCC_array)
avg_RMSE = np.average(RMSE_array)
print('The avg pCC for the 2007 dataset is', avg_pCC)
print('The avg RMSE for the 2007 dataset is', avg_RMSE)