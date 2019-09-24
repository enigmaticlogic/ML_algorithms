import copy
import sys
import pickle
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor as GBT
from sklearn.ensemble import RandomForestRegressor as RF


with open('/mnt/home/storeyd3/Documents/Jobs/features/full_2/all_features_pickled', 'rb') as handle:
    dict_list = pickle.load(handle)

all_proteins_dict_list = [item for sublist in dict_list for item in sublist]

start_index = int(sys.argv[1]) - 1

for index in range(start_index, start_index + 1):
    all_dicts = copy.deepcopy(all_proteins_dict_list)
    popped_dict = all_dicts.pop(index)
    popped_features = popped_dict['features']
    print(popped_features.shape)
    remaining_features = []
    for dictionary in all_dicts:
        remaining_features.append(dictionary['features'])
    current_ID = popped_dict['protein_ID']
    print(current_ID)
    X_test = popped_features[:, 0:59] 
    y_test = popped_features[:, 59]
    popped_features_CA = popped_features[popped_features[:, 60] == 1]
    X_test_CA = popped_features_CA[:, 0:59]
    y_test_CA = popped_features_CA[:, 59]
    remaining_features = np.concatenate(remaining_features, axis=0)
    print(remaining_features.shape)
    X_train = remaining_features[:, 0:59]
    y_train = remaining_features[:, 59]
    
    GBT_reg = GBT(loss = 'ls', n_estimators = 1600, learning_rate = 0.008, \
             max_depth = 4, min_samples_leaf = 9, min_samples_split = 9).fit(X_train, y_train)
    GBT_ypred = GBT_reg.predict(X_test)
    GBT_pCC = stats.pearsonr(GBT_ypred, y_test)[0]
    GBT_ypred_CA = GBT_reg.predict(X_test_CA)
    GBT_pCC_CA = stats.pearsonr(GBT_ypred_CA, y_test_CA)[0]

    RF_reg = RF(n_estimators = 500).fit(X_train, y_train)
    RF_ypred = RF_reg.predict(X_test)
    RF_pCC = stats.pearsonr(RF_ypred, y_test)[0]
    RF_ypred_CA = RF_reg.predict(X_test_CA)
    RF_pCC_CA = stats.pearsonr(RF_ypred_CA, y_test_CA)[0]


GBT_dict = {'protein_ID': current_ID, 'pCC': GBT_pCC, 'pCC_CA': GBT_pCC_CA}
RF_dict = {'protein_ID': current_ID, 'pCC': RF_pCC, 'pCC_CA': RF_pCC_CA}

GBT_pred_filename = '/mnt/home/storeyd3/Documents/Jobs/GBT_ypred_2/pcc'+sys.argv[1]
with open(GBT_pred_filename, 'wb') as handle:
    pickle.dump(GBT_dict, handle)

RF_pred_filename = '/mnt/home/storeyd3/Documents/Jobs/RF_ypred_2/pcc'+sys.argv[1]
with open(RF_pred_filename, 'wb') as handle:
    pickle.dump(RF_dict, handle)

print('success')