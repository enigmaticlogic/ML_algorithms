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

# target_dir = '/mnt/home/storeyd3/Documents/DG_Project/features/v2007_features/LL_v1' #directory to unpickle files from goes here
# output_list = unpickle(target_dir)
# print(len(output_list))
# filename = '/mnt/home/storeyd3/Documents/DG_Project/features/v2007_features/LL_v1/data_dict'#directory to repickle into goes here
# repickle(output_list, filename)

target_dir = '/mnt/home/storeyd3/Documents/DG_Project/features/v2016_features/LL_v1' #directory to unpickle files from goes here
output_list = unpickle(target_dir)
print(len(output_list))
filename = '/mnt/home/storeyd3/Documents/DG_Project/features/v2016_features/LL_v1/data_dict'#directory to repickle into goes here
repickle(output_list, filename)

# target_dir = '/mnt/home/storeyd3/Documents/DG_Project/features/v2007_features/GG_v1' #directory to unpickle files from goes here
# output_list = unpickle(target_dir)
# print(len(output_list))
# filename = '/mnt/home/storeyd3/Documents/DG_Project/features/v2007_features/GG_v1/data_dict'#directory to repickle into goes here
# repickle(output_list, filename)