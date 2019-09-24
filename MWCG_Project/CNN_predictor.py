import copy
import sys
import pickle
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor as GBT
from sklearn.ensemble import RandomForestRegressor as RF
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D, concatenate, Activation, LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import losses


with open('/mnt/home/storeyd3/Documents/Jobs/features/full_2/all_features_pickled', 'rb') as handle:
    dict_list = pickle.load(handle)

all_proteins_dict_list = [item for sublist in dict_list for item in sublist]

def normalize_features(X_train, X_test):
    from sklearn.preprocessing import StandardScaler 
    scaler = StandardScaler() 
    scaler.fit(X_train) 
    X_train_norm = scaler.transform(X_train) 
    X_test_norm = scaler.transform(X_test) 
    return X_train_norm, X_test_norm

start_index = int(sys.argv[1]) - 1

all_dicts = all_proteins_dict_list

for index in range(start_index, start_index + 1):
    # Building the Neural Network here
    K.set_image_data_format('channels_last')

    CNN_in = Input(shape = (8, 30, 3))
    CNN_1 = Conv2D(filters = 14, kernel_size=2)(CNN_in) 
    CNN_act_1 = LeakyReLU(alpha = 0.01)(CNN_1)
    CNN_2 = Conv2D(filters = 16, kernel_size=2)(CNN_act_1) 
    CNN_act_2 = LeakyReLU(alpha = 0.01)(CNN_2)
    CNN_drop_1 = Dropout(0.5)(CNN_act_2)
    CNN_dense_1 = Dense(59)(CNN_drop_1) 
    CNN_act_3 = LeakyReLU(alpha = 0.01)(CNN_dense_1)
    CNN_out = Flatten()(CNN_act_3)
    CNN_model = Model(CNN_in, CNN_out)

    fea_in = Input(shape = (59, ))

    concatenated = concatenate([CNN_out, fea_in])

    merged_dense_1 = Dense(100)(concatenated)
    merged_act_1 = LeakyReLU(alpha = 0.01)(merged_dense_1)
    merged_drop_1 = Dropout(0.5)(merged_act_1)
    merged_dense_2 = Dense(10)(merged_drop_1)
    merged_act_2 = LeakyReLU(alpha = 0.01)(merged_dense_2)
    merged_drop_2 = Dropout(0.25)(merged_act_2)
    merged_out = Dense(1)(merged_drop_2)
    final_model = Model([CNN_in, fea_in], merged_out)

    adam_opt = optimizers.Adam(lr = 0.01)

    popped_dict = all_dicts[index]
    popped_features = popped_dict['features']
    remaining_dicts = [x for i,x in enumerate(all_dicts) if i!=index]
    remaining_features = []
    for dictionary in remaining_dicts:
        remaining_features.append(dictionary['features'])
    current_ID = popped_dict['protein_ID']
    X_test = popped_features[:, 0:59]
    y_test = popped_features[:, 59]
    popped_features_CA = popped_features[popped_features[:,60] == 1]
    X_test_CA = popped_features_CA[:, 0:59]
    y_test_CA = popped_features_CA[:, 59]
    remaining_features = np.concatenate(remaining_features, axis=0)
    X_train = remaining_features[:, 0:59]
    y_train = remaining_features[:, 59]

    popped_CNN_input = popped_dict['CNN_input']
    popped_CNN_input_CA = popped_CNN_input[popped_features[:, 60] == 1]
    remaining_CNN_inputs = []
    for dictionary in remaining_dicts:
        remaining_CNN_inputs.append(dictionary['CNN_input'])
    CNN_input = np.concatenate(remaining_CNN_inputs, axis = 0)

    CNN_input_flat = np.reshape(CNN_input, (-1, 720))
    popped_CNN_input_flat = np.reshape(popped_CNN_input, (-1, 720))
    popped_CNN_input_CA_flat = np.reshape(popped_CNN_input_CA, (-1, 720))

    CNN_input_flat_norm, popped_CNN_input_flat_norm = normalize_features(CNN_input_flat, popped_CNN_input_flat)

    CNN_input_flat_norm, popped_CNN_input_CA_flat_norm = normalize_features(CNN_input_flat, popped_CNN_input_CA_flat)

    CNN_input_norm = np.reshape(CNN_input_flat_norm, (-1, 8, 30, 3))
    popped_CNN_input_norm = np.reshape(popped_CNN_input_flat_norm, (-1, 8, 30, 3))
    popped_CNN_input_CA_norm = np.reshape(popped_CNN_input_CA_flat_norm, (-1, 8, 30, 3))  

    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    X_train_norm, X_test_CA_norm = normalize_features(X_train, X_test_CA)
    
    final_model.compile(loss = 'mean_absolute_error', optimizer = adam_opt)
    final_model.fit([CNN_input_norm, X_train_norm], y_train, epochs = 100 , batch_size = 100) 
    CNN_ypred = final_model.predict([popped_CNN_input_norm, X_test_norm])
    CNN_ypred = CNN_ypred.reshape(y_test.shape)
    CNN_pCC = stats.pearsonr(CNN_ypred, y_test)[0]
    CNN_ypred_CA = final_model.predict([popped_CNN_input_CA_norm, X_test_CA_norm])
    CNN_ypred_CA = CNN_ypred_CA.reshape(y_test_CA.shape)
    CNN_pCC_CA = stats.pearsonr(CNN_ypred_CA, y_test_CA)[0]
    print(CNN_pCC)
    print(index)
    CNN_dict = {'protein_ID': current_ID, 'pCC': CNN_pCC, 'pCC_CA': CNN_pCC_CA}

CNN_pred_filename = '/mnt/home/storeyd3/Documents/Jobs/CNN_ypred/pcc'+sys.argv[1]
with open(CNN_pred_filename, 'wb') as handle:
    pickle.dump(CNN_dict, handle)

print('success')