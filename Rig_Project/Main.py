import numpy as np
import pickle
import os
import sys
import pandas as pd
from scipy import spatial as sp

from Utility.Classes import atom
# Has attributes pos, heavy_type, atom_name, Rval, res, B_factor
from Utility.PDB_Reader import read_pdb
# Input: pdb file   Output: atom list
from Utility.SDF_Reader import read_sdf
# Input: sdf file   Output: atom list
from Utility.Index_Reader import read_index
# Input: index file   Output: pandas df containing ID's and energies
from Utility.Split_Atoms import split_atoms
# Input: atom list  Output: dictionary of lists with key ['C', 'N', 'O', 'S', 'H', 'P', 'F', 'Cl', 'Br', 'I']
from Utility.Cutoff import get_cutoff_pairs
# Input: atom list 1, atom list 2, cutoff distance  Output: list of atom pairs (each a 2 element list)

#------------------------------Load Data-------------------------------------#
# 2007 dataset
# protein_dir = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2007' # Dir to get PDB's from
# protein_ID_list = sorted(os.listdir(protein_dir))
# refined_energy_file = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2007_extra/v2007_refine_list.csv'
# core_energy_file = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2007_extra/v2007_core_list.csv'
# protein_refined_energy_df = pd.read_csv(refined_energy_file)
# protein_core_energy_df = pd.read_csv(core_energy_file)
# protein_refined_energy_array = protein_refined_energy_df.values
# protein_core_energy_array = protein_core_energy_df.values

# protein_train_dict = {'ID_list': [], 'pdb_file_list': [], 'sdf_file_list': [], 'binding_energy_list': []}
# protein_test_dict = {'ID_list': [], 'pdb_file_list': [], 'sdf_file_list': [], 'binding_energy_list': []}
# for ID in protein_ID_list:
#     if ID in protein_refined_energy_array[:,0]:
#         protein_train_dict['ID_list'].append(ID)
#         protein_train_dict['pdb_file_list'].append(protein_dir+'/'+ID+'/'+ID+'_protein.pdb')
#         protein_train_dict['sdf_file_list'].append(protein_dir+'/'+ID+'/'+ID+'_ligand.sdf')
#         protein_train_dict['binding_energy_list'].append\
#           (protein_refined_energy_array[protein_refined_energy_array[:,0] == ID][0,1])
#     if ID in protein_core_energy_array[:,0]:
#         protein_test_dict['ID_list'].append(ID)
#         protein_test_dict['pdb_file_list'].append(protein_dir+'/'+ID+'/'+ID+'_protein.pdb')
#         protein_test_dict['sdf_file_list'].append(protein_dir+'/'+ID+'/'+ID+'_ligand.sdf')
#         protein_test_dict['binding_energy_list'].append\
#           (protein_core_energy_array[protein_core_energy_array[:,0] == ID][0,1])

# 2013 dataset
refined_protein_dir = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2013_refined' # Dir to get PDB's from
refined_ID_list = sorted(os.listdir(refined_protein_dir))
core_protein_dir = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2013_core' # Dir to get PDB's from
core_ID_list = sorted(os.listdir(core_protein_dir))

refined_index_filename = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2013_extra/INDEX_refined_data.2013'
core_index_filename = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2013_extra/INDEX_core_data.2013'
protein_refined_energy_df = read_index(refined_index_filename)
protein_core_energy_df = read_index(core_index_filename)
protein_refined_energy_array = protein_refined_energy_df.values
protein_core_energy_array = protein_core_energy_df.values

protein_train_dict = {'ID_list': [], 'pdb_file_list': [], 'sdf_file_list': [], 'binding_energy_list': []}
protein_test_dict = {'ID_list': [], 'pdb_file_list': [], 'sdf_file_list': [], 'binding_energy_list': []}

for ID in refined_ID_list:
    protein_train_dict['ID_list'].append(ID)
    protein_train_dict['pdb_file_list'].append(refined_protein_dir+'/'+ID+'/'+ID+'_protein.pdb')
    protein_train_dict['sdf_file_list'].append(refined_protein_dir+'/'+ID+'/'+ID+'_ligand.sdf')
    protein_train_dict['binding_energy_list'].append\
      (protein_refined_energy_array[protein_refined_energy_array[:,0] == ID][0,1])

for ID in core_ID_list:
    protein_test_dict['ID_list'].append(ID)
    protein_test_dict['pdb_file_list'].append(core_protein_dir+'/'+ID+'/'+ID+'_protein.pdb')
    protein_test_dict['sdf_file_list'].append(core_protein_dir+'/'+ID+'/'+ID+'_ligand.sdf')
    protein_test_dict['binding_energy_list'].append\
      (protein_core_energy_array[protein_core_energy_array[:,0] == ID][0,1])

#------------------------------Classes-------------------------------------#
class feature_compiler:
    def __init__(self, pdb_file, sdf_file):
        self.pdb_file = pdb_file
        self.sdf_file = sdf_file

    def get_features(self, kernels, betas, taus, cutoffs):
        protein_atoms_list = read_pdb(self.pdb_file)
        ligand_atoms_list = read_sdf(self.sdf_file)
        protein_split_dict = split_atoms(protein_atoms_list)
        ligand_split_dict = split_atoms(ligand_atoms_list)

        protein_heavy_types = ['C', 'N', 'O', 'S']
        ligand_heavy_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        features = np.zeros((len(kernels), len(ligand_heavy_types), len(protein_heavy_types)))
        protein_count = 0
        for protein_index in protein_heavy_types:
            ligand_count = 0
            for ligand_index in ligand_heavy_types:
                for param_index in range(len(kernels)):
                    current_cutoff_pairs = get_cutoff_pairs(protein_split_dict[protein_index], \
                      ligand_split_dict[ligand_index], cutoffs[param_index])
                    kernel_output_array = np.zeros((len(kernels), len(current_cutoff_pairs)))
                    for pair_index in range(len(current_cutoff_pairs)):
                        vdr_1 = get_vdr(current_cutoff_pairs[pair_index][0].heavy_type) 
                        vdr_2 = get_vdr(current_cutoff_pairs[pair_index][1].heavy_type) 
                        current_eta = taus[param_index]*(vdr_1 + vdr_2)
                        current_dist = np.linalg.norm(current_cutoff_pairs[pair_index][0].pos - \
                          current_cutoff_pairs[pair_index][1].pos)
                        current_kernel_output = kernel_function(kernels[param_index], current_dist,\
                          betas[param_index], current_eta)
                        kernel_output_array[param_index, pair_index] = current_kernel_output
                    features[param_index, ligand_count, protein_count] = sum(kernel_output_array[param_index, :])
                ligand_count += 1
            protein_count += 1

        features = features.reshape(len(kernels)*36)
        return features


#------------------------------Functions-------------------------------------#

def kernel_function(type, dist, beta, eta):
    if type == 'gaussian':
        return np.exp(-(dist/eta)**beta)
    elif type == 'lorentz':
        return 1/(1+(dist/eta)**beta)
    else:
        print('error in kernel name')

def get_vdr(heavy_atom_type):
    ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    if heavy_atom_type == 'C':
        vdr = 1.7
    elif heavy_atom_type == 'N':
        vdr = 1.55
    elif heavy_atom_type == 'O':
        vdr = 1.52
    elif heavy_atom_type == 'S':
        vdr = 1.8
    elif heavy_atom_type == 'P':
        vdr = 1.8
    elif heavy_atom_type == 'F':
        vdr = 1.47
    elif heavy_atom_type == 'Cl':
        vdr = 1.75
    elif heavy_atom_type == 'Br':
        vdr = 1.85
    elif heavy_atom_type == 'I':
        vdr = 1.98

    return vdr

#------------------------------Execution-------------------------------------#
# 2007 parameters
# kernels = ['gaussian']
# betas = [2.5]
# taus = [1]
# cutoffs = [12]
# num_feas = len(kernels)*36

# 2013 parameters
kernels = ['lorentz']
betas = [6]
taus = [1.5]
cutoffs = [9]
num_feas = len(kernels)*36

num_train_samples = len(protein_train_dict['ID_list'])
X_train = np.zeros((num_train_samples, num_feas))
y_train = np.zeros((num_train_samples,))

for index in range(num_train_samples):
    fea_compiler = feature_compiler(protein_train_dict['pdb_file_list'][index], \
      protein_train_dict['sdf_file_list'][index])
    features = fea_compiler.get_features(kernels, betas, taus, cutoffs)
    X_train[index, :] = features
    y_train[index] = protein_train_dict['binding_energy_list'][index]
    print(100*index/num_train_samples, 'percent done with training set')

num_test_samples = len(protein_test_dict['ID_list'])
X_test = np.zeros((num_test_samples, num_feas))
y_test = np.zeros((num_test_samples,))

for index in range(num_test_samples):
    fea_compiler = feature_compiler(protein_test_dict['pdb_file_list'][index], \
      protein_test_dict['sdf_file_list'][index])
    features = fea_compiler.get_features(kernels, betas, taus, cutoffs)
    X_test[index, :] = features
    y_test[index] = protein_test_dict['binding_energy_list'][index]
    print(100*index/num_test_samples, 'percent done with test set')

# 2007 dataset
# X_train_filename = '/mnt/home/storeyd3/Documents/FRI_Project/v2007_features/'+'X_train' 
# y_train_filename = '/mnt/home/storeyd3/Documents/FRI_Project/v2007_features/'+'y_train' 
# X_test_filename = '/mnt/home/storeyd3/Documents/FRI_Project/v2007_features/'+'X_test' 
# y_test_filename = '/mnt/home/storeyd3/Documents/FRI_Project/v2007_features/'+'y_test' 

# 2013 dataset
X_train_filename = '/mnt/home/storeyd3/Documents/FRI_Project/v2013_features/'+'X_train' 
y_train_filename = '/mnt/home/storeyd3/Documents/FRI_Project/v2013_features/'+'y_train' 
X_test_filename = '/mnt/home/storeyd3/Documents/FRI_Project/v2013_features/'+'X_test' 
y_test_filename = '/mnt/home/storeyd3/Documents/FRI_Project/v2013_features/'+'y_test' 

with open(X_train_filename, 'wb') as file:
    pickle.dump(X_train, file)
with open(y_train_filename, 'wb') as file:
    pickle.dump(y_train, file)
with open(X_test_filename, 'wb') as file:
    pickle.dump(X_test, file)
with open(y_test_filename, 'wb') as file:
    pickle.dump(y_test, file)
