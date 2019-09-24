import copy
import sys
import os
import pickle
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor as GBT
from sklearn.ensemble import RandomForestRegressor as RF


with open('/mnt/home/storeyd3/Documents/Jobs/GBT_ypred_2/all_GBT_pcc_pickled', 'rb') as handle:
    GBT_dict_list = pickle.load(handle)

with open('/mnt/home/storeyd3/Documents/Jobs/RF_ypred_2/all_RF_pcc_pickled', 'rb') as handle:
    RF_dict_list = pickle.load(handle)

with open('/mnt/home/storeyd3/Documents/Jobs/CNN_ypred/all_CNN_pcc_pickled', 'rb') as handle:
    CNN_dict_list = pickle.load(handle) 

small_dir = '/mnt/home/storeyd3/Documents/Datasets/park_small_trim' 
medium_dir = '/mnt/home/storeyd3/Documents/Datasets/park_medium_trim' 
large_dir = '/mnt/home/storeyd3/Documents/Datasets/park_large_trim' 

for dictionary in GBT_dict_list:
    print('The GBT pCC for the protein ', dictionary['protein_ID'], 'is ', dictionary['pCC'])
    print('The GBT C Alpha only pCC for the protein ', dictionary['protein_ID'], 'is ', dictionary['pCC_CA'])

small_GBT_pCC_list = []
medium_GBT_pCC_list = []
large_GBT_pCC_list = []
full_GBT_pCC_list = []
small_GBT_pCC_list_CA = []
medium_GBT_pCC_list_CA = []
large_GBT_pCC_list_CA = []
full_GBT_pCC_list_CA = []

for dictionary in GBT_dict_list:
    if dictionary['protein_ID'] in os.listdir(small_dir):
        small_GBT_pCC_list.append(dictionary['pCC'])
        small_GBT_pCC_list_CA.append(dictionary['pCC_CA'])
    if dictionary['protein_ID'] in os.listdir(medium_dir):
        medium_GBT_pCC_list.append(dictionary['pCC'])
        medium_GBT_pCC_list_CA.append(dictionary['pCC_CA'])
    if dictionary['protein_ID'] in os.listdir(large_dir):
        large_GBT_pCC_list.append(dictionary['pCC'])
        large_GBT_pCC_list_CA.append(dictionary['pCC_CA'])
    full_GBT_pCC_list.append(dictionary['pCC'])
    full_GBT_pCC_list_CA.append(dictionary['pCC_CA'])

small_GBT_pCC_array = np.array(small_GBT_pCC_list)
small_GBT_avg_pCC = np.average(small_GBT_pCC_array)
small_GBT_pCC_array_CA = np.array(small_GBT_pCC_list_CA)
small_GBT_avg_pCC_CA = np.average(small_GBT_pCC_array_CA)

medium_GBT_pCC_array = np.array(medium_GBT_pCC_list)
medium_GBT_avg_pCC = np.average(medium_GBT_pCC_array)
medium_GBT_pCC_array_CA = np.array(medium_GBT_pCC_list_CA)
medium_GBT_avg_pCC_CA = np.average(medium_GBT_pCC_array_CA)

large_GBT_pCC_array = np.array(large_GBT_pCC_list)
large_GBT_avg_pCC = np.average(large_GBT_pCC_array)
large_GBT_pCC_array_CA = np.array(large_GBT_pCC_list_CA)
large_GBT_avg_pCC_CA = np.average(large_GBT_pCC_array_CA)

full_GBT_pCC_array = np.array(full_GBT_pCC_list)
full_GBT_avg_pCC = np.average(full_GBT_pCC_array)
full_GBT_pCC_array_CA = np.array(full_GBT_pCC_list_CA)
full_GBT_avg_pCC_CA = np.average(full_GBT_pCC_array_CA)

print('GBT Accuracies:')
print('Small GBT C Alpha only average Pearson correlation coefficient: ', small_GBT_avg_pCC_CA)
print('Medium GBT C Alpha only average Pearson correlation coefficient: ', medium_GBT_avg_pCC_CA)
print('Large GBT C Alpha only average Pearson correlation coefficient: ', large_GBT_avg_pCC_CA)
print('Full GBT C Alpha only average Pearson correlation coefficient: ', full_GBT_avg_pCC_CA)
print(' ')
print('Small GBT average Pearson correlation coefficient: ', small_GBT_avg_pCC)
print('Medium GBT average Pearson correlation coefficient: ', medium_GBT_avg_pCC)
print('Large GBT average Pearson correlation coefficient: ', large_GBT_avg_pCC)
print('Full GBT average Pearson correlation coefficient: ', full_GBT_avg_pCC)
print(' ')

for dictionary in RF_dict_list:
    print('The RF pCC for the protein ', dictionary['protein_ID'], 'is ', dictionary['pCC'])
    print('The RF C Alpha only pCC for the protein ', dictionary['protein_ID'], 'is ', dictionary['pCC_CA'])

small_RF_pCC_list = []
medium_RF_pCC_list = []
large_RF_pCC_list = []
full_RF_pCC_list = []
small_RF_pCC_list_CA = []
medium_RF_pCC_list_CA = []
large_RF_pCC_list_CA = []
full_RF_pCC_list_CA = []

for dictionary in RF_dict_list:
    if dictionary['protein_ID'] in os.listdir(small_dir):
        small_RF_pCC_list.append(dictionary['pCC'])
        small_RF_pCC_list_CA.append(dictionary['pCC_CA'])
    if dictionary['protein_ID'] in os.listdir(medium_dir):
        medium_RF_pCC_list.append(dictionary['pCC'])
        medium_RF_pCC_list_CA.append(dictionary['pCC_CA'])
    if dictionary['protein_ID'] in os.listdir(large_dir):
        large_RF_pCC_list.append(dictionary['pCC'])
        large_RF_pCC_list_CA.append(dictionary['pCC_CA'])
    full_RF_pCC_list.append(dictionary['pCC'])
    full_RF_pCC_list_CA.append(dictionary['pCC_CA'])

small_RF_pCC_array = np.array(small_RF_pCC_list)
small_RF_avg_pCC = np.average(small_RF_pCC_array)
small_RF_pCC_array_CA = np.array(small_RF_pCC_list_CA)
small_RF_avg_pCC_CA = np.average(small_RF_pCC_array_CA)

medium_RF_pCC_array = np.array(medium_RF_pCC_list)
medium_RF_avg_pCC = np.average(medium_RF_pCC_array)
medium_RF_pCC_array_CA = np.array(medium_RF_pCC_list_CA)
medium_RF_avg_pCC_CA = np.average(medium_RF_pCC_array_CA)

large_RF_pCC_array = np.array(large_RF_pCC_list)
large_RF_avg_pCC = np.average(large_RF_pCC_array)
large_RF_pCC_array_CA = np.array(large_RF_pCC_list_CA)
large_RF_avg_pCC_CA = np.average(large_RF_pCC_array_CA)

full_RF_pCC_array = np.array(full_RF_pCC_list)
full_RF_avg_pCC = np.average(full_RF_pCC_array)
full_RF_pCC_array_CA = np.array(full_RF_pCC_list_CA)
full_RF_avg_pCC_CA = np.average(full_RF_pCC_array_CA)

print('RF Accuracies:')
print('Small RF C Alpha only average Pearson correlation coefficient: ', small_RF_avg_pCC_CA)
print('Medium RF C Alpha only average Pearson correlation coefficient: ', medium_RF_avg_pCC_CA)
print('Large RF C Alpha only average Pearson correlation coefficient: ', large_RF_avg_pCC_CA)
print('Full RF C Alpha only average Pearson correlation coefficient: ', full_RF_avg_pCC_CA)
print(' ')
print('Small RF average Pearson correlation coefficient: ', small_RF_avg_pCC)
print('Medium RF average Pearson correlation coefficient: ', medium_RF_avg_pCC)
print('Large RF average Pearson correlation coefficient: ', large_RF_avg_pCC)
print('Full RF average Pearson correlation coefficient: ', full_RF_avg_pCC)
print(' ')

for dictionary in CNN_dict_list:
    print('The CNN pCC for the protein ', dictionary['protein_ID'], 'is ', dictionary['pCC'])
    print('The CNN C Alpha only pCC for the protein ', dictionary['protein_ID'], 'is ', dictionary['pCC_CA'])

small_CNN_pCC_list = []
medium_CNN_pCC_list = []
large_CNN_pCC_list = []
full_CNN_pCC_list = []
small_CNN_pCC_list_CA = []
medium_CNN_pCC_list_CA = []
large_CNN_pCC_list_CA = []
full_CNN_pCC_list_CA = []

for dictionary in CNN_dict_list:
    if dictionary['protein_ID'] in os.listdir(small_dir):
        small_CNN_pCC_list.append(dictionary['pCC'])
        small_CNN_pCC_list_CA.append(dictionary['pCC_CA'])
    if dictionary['protein_ID'] in os.listdir(medium_dir):
        medium_CNN_pCC_list.append(dictionary['pCC'])
        medium_CNN_pCC_list_CA.append(dictionary['pCC_CA'])
    if dictionary['protein_ID'] in os.listdir(large_dir):
        large_CNN_pCC_list.append(dictionary['pCC'])
        large_CNN_pCC_list_CA.append(dictionary['pCC_CA'])
    full_CNN_pCC_list.append(dictionary['pCC'])
    full_CNN_pCC_list_CA.append(dictionary['pCC_CA'])

small_CNN_pCC_array = np.array(small_CNN_pCC_list)
small_CNN_avg_pCC = np.average(small_CNN_pCC_array)
small_CNN_pCC_array_CA = np.array(small_CNN_pCC_list_CA)
small_CNN_avg_pCC_CA = np.average(small_CNN_pCC_array_CA)

medium_CNN_pCC_array = np.array(medium_CNN_pCC_list)
medium_CNN_avg_pCC = np.average(medium_CNN_pCC_array)
medium_CNN_pCC_array_CA = np.array(medium_CNN_pCC_list_CA)
medium_CNN_avg_pCC_CA = np.average(medium_CNN_pCC_array_CA)

large_CNN_pCC_array = np.array(large_CNN_pCC_list)
large_CNN_avg_pCC = np.average(large_CNN_pCC_array)
large_CNN_pCC_array_CA = np.array(large_CNN_pCC_list_CA)
large_CNN_avg_pCC_CA = np.average(large_CNN_pCC_array_CA)

full_CNN_pCC_array = np.array(full_CNN_pCC_list)
full_CNN_avg_pCC = np.average(full_CNN_pCC_array)
full_CNN_pCC_array_CA = np.array(full_CNN_pCC_list_CA)
full_CNN_pCC_array_CA = full_CNN_pCC_array_CA[np.isfinite(full_CNN_pCC_array_CA)]
full_CNN_avg_pCC_CA = np.average(full_CNN_pCC_array_CA)

print('CNN Accuracies:')
print('Small CNN C Alpha only average Pearson correlation coefficient: ', small_CNN_avg_pCC_CA)
print('Medium CNN C Alpha only average Pearson correlation coefficient: ', medium_CNN_avg_pCC_CA)
print('Large CNN C Alpha only average Pearson correlation coefficient: ', large_CNN_avg_pCC_CA)
print('Full CNN C Alpha only average Pearson correlation coefficient: ', full_CNN_avg_pCC_CA)
print(' ')
print('Small CNN average Pearson correlation coefficient: ', small_CNN_avg_pCC)
print('Medium CNN average Pearson correlation coefficient: ', medium_CNN_avg_pCC)
print('Large CNN average Pearson correlation coefficient: ', large_CNN_avg_pCC)
print('Full CNN average Pearson correlation coefficient: ', full_CNN_avg_pCC)
print(' ')