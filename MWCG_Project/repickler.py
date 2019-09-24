import pickle
import numpy as np
import os
'''
target_dir = directory containing files to be unpickled here
'''
def unpickle(target_dir):
	output_list = []
	for file in sorted(os.listdir(target_dir)):
		with open(target_dir+'/'+file, 'rb') as handle:
			one_output = pickle.load(handle)
		output_list.append(one_output)
	return output_list

def repickle(output_list, filename):
	with open(filename, 'wb') as handle:
		pickle.dump(output_list, handle)

target_dir = '/mnt/home/storeyd3/Documents/Jobs/features/full' #directory to unpickle files from goes here
output_list = unpickle(target_dir)
print(len(output_list))
filename = '/mnt/home/storeyd3/Documents/Jobs/features/full/all_features_pickled'#directory to repickle into goes here
repickle(output_list, filename)

# target_dir = '/mnt/home/storeyd3/Documents/Jobs/CNN_ypred' #directory to unpickle files from goes here
# output_list = unpickle(target_dir)
# print(len(output_list))
# filename = '/mnt/home/storeyd3/Documents/Jobs/CNN_ypred/all_CNN_pcc_pickled'#directory to repickle into goes here
# repickle(output_list, filename)
