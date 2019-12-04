import numpy as np
import pickle
import os
import sys
import math
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
from Utility.Cutoff import get_cutoff_list
# Input: protein atom, ligand atom list, cutoff distance  Output: list of ligand atoms outside the cutoff
from Utility.Cutoff import get_binding_site
# Input: protein atom list, ligand atom list, cutoff distance  Output: list of protein atoms within cutoff

#------------------------------Load Data-------------------------------------#
# 2007 dataset
# protein_dir = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2007' # Dir to get PDB's from
# protein_ID_list = sorted(os.listdir(protein_dir))
# refined_energy_file = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2007_extra/v2007_refine_list.csv'
# core_energy_file = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2007_extra/v2007_core_list.csv'
# protein_refined_energy_df = pd.read_csv(refined_energy_file, header=None)
# protein_core_energy_df = pd.read_csv(core_energy_file, header=None)
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

# 2016 dataset
protein_dir = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2016/refined_set' # Dir to get PDB's from
protein_ID_list = sorted(os.listdir(protein_dir))
print(protein_ID_list)
refined_energy_file = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2016_extra/v2016_refine_list.csv'
core_energy_file = '/mnt/home/storeyd3/Documents/Datasets/PDBBind_v2016_extra/v2016_core_list.csv'
protein_refined_energy_df = pd.read_csv(refined_energy_file, header=None)
protein_core_energy_df = pd.read_csv(core_energy_file, header=None)
protein_refined_energy_array = protein_refined_energy_df.values
protein_core_energy_array = protein_core_energy_df.values

protein_train_dict = {'ID_list': [], 'pdb_file_list': [], 'sdf_file_list': [], 'binding_energy_list': []}
protein_test_dict = {'ID_list': [], 'pdb_file_list': [], 'sdf_file_list': [], 'binding_energy_list': []}
for ID in protein_ID_list:
    if ID in protein_refined_energy_array[:,0]:
        protein_train_dict['ID_list'].append(ID)
        protein_train_dict['pdb_file_list'].append(protein_dir+'/'+ID+'/'+ID+'_protein.pdb')
        protein_train_dict['sdf_file_list'].append(protein_dir+'/'+ID+'/'+ID+'_ligand.sdf')
        protein_train_dict['binding_energy_list'].append\
          (protein_refined_energy_array[protein_refined_energy_array[:,0] == ID][0,1])
    if ID in protein_core_energy_array[:,0]:
        protein_test_dict['ID_list'].append(ID)
        protein_test_dict['pdb_file_list'].append(protein_dir+'/'+ID+'/'+ID+'_protein.pdb')
        protein_test_dict['sdf_file_list'].append(protein_dir+'/'+ID+'/'+ID+'_ligand.sdf')
        protein_test_dict['binding_energy_list'].append\
          (protein_core_energy_array[protein_core_energy_array[:,0] == ID][0,1])

dataset = '2016'

#------------------------------Classes-------------------------------------#
class feature_compiler:
    def __init__(self, pdb_file, sdf_file):
        self.pdb_file = pdb_file
        self.sdf_file = sdf_file

    def get_features(self, kernels, betas, taus, curv_type):
        protein_atoms_list = read_pdb(self.pdb_file)
        ligand_atoms_list = read_sdf(self.sdf_file, dataset)
        protein_atoms_list = get_binding_site(protein_atoms_list, ligand_atoms_list, 20)

        protein_split_dict = split_atoms(protein_atoms_list)
        ligand_split_dict = split_atoms(ligand_atoms_list)

        protein_heavy_types = ['C', 'N', 'O', 'S']
        ligand_heavy_types = ['C', 'N', 'O', 'S', 'H', 'P', 'F', 'Cl', 'Br', 'I']
        sigma = 0.5
        features = np.zeros((len(kernels), len(ligand_heavy_types), len(protein_heavy_types), 10))
        for protein_index in range(len(protein_heavy_types)):
            vdr_1 = get_vdr(protein_heavy_types[protein_index])
            for ligand_index in range(len(ligand_heavy_types)):
                vdr_2 = get_vdr(ligand_heavy_types[ligand_index])
                current_cutoff = vdr_1 + vdr_2 + sigma
                p_heavy_atom_list = protein_split_dict[protein_heavy_types[protein_index]]
                l_heavy_atom_list = ligand_split_dict[ligand_heavy_types[ligand_index]]
                for param_index in range(len(kernels)):   
                    current_eta = taus[param_index]*(vdr_1 + vdr_2) 
                    curvature_list = []
                    for p_heavy_index in range(len(p_heavy_atom_list)): 
                        current_p_atom = p_heavy_atom_list[p_heavy_index]
                        l_cutoff_atoms = get_cutoff_list(current_p_atom, l_heavy_atom_list, current_cutoff)
                        partial_curvature_list = []
                        for l_heavy_index in range(len(l_cutoff_atoms)):
                            current_l_atom = l_cutoff_atoms[l_heavy_index]         
                            current_partial_curvature = get_curvature(current_p_atom.pos, current_l_atom.pos, \
                              kernels[param_index], current_eta, betas[param_index], curv_type[param_index])
                            partial_curvature_list.append(current_partial_curvature)
                        current_curvature = sum(partial_curvature_list)
                        curvature_list.append(current_curvature)

                    abs_curvature_list = list(map(abs, curvature_list))
                    current_features = np.zeros(10)

                    if len(curvature_list) > 0:
                        current_features[0] = sum(curvature_list)
                        current_features[1] = min(curvature_list)
                        current_features[2] = max(curvature_list)
                        current_features[3] = np.mean(curvature_list)
                        current_features[4] = np.std(curvature_list)

                        current_features[5] = sum(abs_curvature_list)
                        current_features[6] = min(abs_curvature_list)
                        current_features[7] = max(abs_curvature_list)
                        current_features[8] = np.mean(abs_curvature_list)
                        current_features[9] = np.std(abs_curvature_list)

                    features[param_index, ligand_index, protein_index, :] = current_features

        features = features.reshape(len(kernels)*len(protein_heavy_types)*len(ligand_heavy_types)*10)
        return features


#------------------------------Functions-------------------------------------#

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
    elif heavy_atom_type == 'H':
        vdr = 1.2

    return vdr

def get_curvature(pos_1, pos_2, kernel, eta, beta, curv_type):
    x = pos_1[0]
    y = pos_1[1]
    z = pos_1[2]
    a = pos_2[0]
    b = pos_2[1]
    c = pos_2[2] 
    A = (np.linalg.norm(pos_1 - pos_2)/eta)**beta
    B = (a - x)**2 + (b - y)**2 + (c - z)**2
    k = beta # For readability
    
    if kernel == 'lorentz':
        sub0 = A + 1
        sub1 = (B**2*sub0**3)
        sub2 = (a - x)**2
        sub3 = (b - y)**2
        sub4 = (c - z)**2
        sub5 = (B*sub0**2)   
        sub6 = (k*A + 2*A - k + 2)/sub1
        sub7 = (k - 1)*sub0
        sub8 = B*sub0
        sub9 = k*A

        Dx = sub9*(a - x)/sub5
        Dy = sub9*(b - y)/sub5
        Dz = sub9*(c - z)/sub5
        Dxx = sub9*(2*sub9*sub2 - sub7*sub2 + sub2*sub0 - sub8)/ \
          sub1
        Dyy = sub9*(2*sub9*sub3 - sub7*sub3 + sub3*sub0 - sub8)/ \
          sub1
        Dzz = sub9*(2*sub9*sub4 - sub7*sub4 + sub4*sub0 - sub8)/ \
          sub1
        Dxy = sub9*(a - x)*(b - y)*sub6
        Dxz = sub9*(a - x)*(c - z)*sub6
        Dyz = sub9*(b - y)*(c - z)*sub6
    elif kernel == 'gaussian':
        sub0 = k*np.exp(-A)*A
        sub1 = (k*A - k + 2)/(B**2)
        Dx = (a - x)*sub0/(B)
        Dy = (b - y)*sub0/(B)
        Dz = (c - z)*sub0/(B)
        Dxx = sub0*(a**2*k*A - a**2*k + a**2 + k*x**2*A - 2*a*k*x*A + 2*a*k*x - 2*a*x \
            - b**2 + 2*b*y - c**2 + 2*c*z - k*x**2 + x**2 - y**2 - z**2)/(B**2)
        Dyy = -sub0*(a**2 - b**2*k*A - k*y**2*A + 2*b*k*y*A - 2*a*x + b**2*k - b**2 - 2*b*k*y \
          + 2*b*y + c**2 - 2*c*z + k*y**2 + x**2 - y**2 + z**2)/(B**2)
        Dzz = -sub0*(a**2 - c**2*k*A - k*z**2*A + 2*c*k*z*A - 2*a*x + b**2 - 2*b*y + c**2*k \
          - c**2 - 2*c*k*z + 2*c*z + k*z**2 + x**2 + y**2 - z**2)/(B**2)
        Dxy = (a - x)*(b - y)*sub0*sub1
        Dxz = (a - x)*(c - z)*sub0*sub1
        Dyz = (b - y)*(c - z)*sub0*sub1

    g = Dx**2 + Dy**2 + Dz**2

    if curv_type == 'gaussian':
        curvature = (1/g**2)*(2*Dx*Dy*Dxz*Dyz + 2*Dx*Dz*Dxy*Dyz + 2*Dy*Dz*Dxy*Dxz - 2*Dx*Dz*Dxz*Dyy \
          - 2*Dy*Dz*Dxx*Dyz - 2*Dx*Dy*Dxy*Dzz + Dz**2*Dxx*Dyy + Dx**2*Dyy*Dzz + Dy**2*Dxx*Dzz \
            - Dx**2*Dyz**2 - Dy**2*Dxz**2 - Dz**2*Dxy**2)
    elif curv_type == 'mean':
        curvature = (1/(2*g**(3/2)))*(2*Dx*Dy*Dxy + 2*Dx*Dz*Dxz + 2*Dy*Dz*Dyz \
          - Dxx*(Dy**2 + Dz**2) - Dyy*(Dx**2 + Dz**2) - Dzz*(Dx**2 + Dy**2))
    else:
        g_curv = (1/g**2)*(2*Dx*Dy*Dxz*Dyz + 2*Dx*Dz*Dxy*Dyz + 2*Dy*Dz*Dxy*Dxz - 2*Dx*Dz*Dxz*Dyy \
          - 2*Dy*Dz*Dxx*Dyz - 2*Dx*Dy*Dxy*Dzz + Dz**2*Dxx*Dyy + Dx**2*Dyy*Dzz + Dy**2*Dxx*Dzz \
            - Dx**2*Dyz**2 - Dy**2*Dxz**2 - Dz**2*Dxy**2)
        m_curv = (1/(2*g**(3/2)))*(2*Dx*Dy*Dxy + 2*Dx*Dz*Dxz + 2*Dy*Dz*Dyz \
          - Dxx*(Dy**2 + Dz**2) - Dyy*(Dx**2 + Dz**2) - Dzz*(Dx**2 + Dy**2))
        if curv_type == 'min':
            curvature = m_curv - (m_curv**2 - g_curv)**(1/2)
        elif curv_type == 'max': 
            curvature = m_curv + (m_curv**2 - g_curv)**(1/2)

    return curvature

#------------------------------Execution-------------------------------------#
# 2007 parameters
kernels = ['lorentz', 'lorentz']
betas = [4.5, 5.5]
taus = [2.5, 5]
curv_types = ['mean', 'mean']
num_feas = len(kernels)*400
parameters = [kernels, betas, taus, curv_types]

# 2013 parameters
# kernels = ['lorentz']
# betas = [6]
# taus = [1.5]
# cutoffs = [9]
# num_feas = len(kernels)*36

num_train_samples = len(protein_train_dict['ID_list'])
num_test_samples = len(protein_test_dict['ID_list'])
print(num_train_samples)
print(num_test_samples)
# Next time try splitting up lists instead of indices
sample_split = 75 # Number of pieces to split samples into for faster computing
num_train_indices = math.floor(num_train_samples/sample_split)
num_test_indices = math.floor(num_test_samples/sample_split)

start_index = int(sys.argv[1]) - 1
if int(sys.argv[1]) == sample_split:
    train_batch_range = range(start_index*num_train_indices, num_train_samples)
    test_batch_range = range(start_index*num_test_indices, num_test_samples)
else:
    train_batch_range = range(start_index*num_train_indices, (start_index + 1)*num_train_indices)
    test_batch_range = range(start_index*num_test_indices, (start_index + 1)*num_test_indices)

train_batch_size = len(train_batch_range)
test_batch_size = len(test_batch_range)

X_train = np.zeros((train_batch_size, num_feas))
y_train = np.zeros((train_batch_size,))
X_test = np.zeros((test_batch_size, num_feas))
y_test = np.zeros((test_batch_size,))

train_index = 0
for index in train_batch_range:
    fea_compiler = feature_compiler(protein_train_dict['pdb_file_list'][index], \
      protein_train_dict['sdf_file_list'][index])
    features = fea_compiler.get_features(kernels, betas, taus, curv_types)
    X_train[train_index, :] = features
    y_train[train_index] = protein_train_dict['binding_energy_list'][index]
    # print(100*index/num_train_samples, 'percent done with training set')
    train_index += 1

test_index = 0
for index in test_batch_range:
    fea_compiler = feature_compiler(protein_test_dict['pdb_file_list'][index], \
      protein_test_dict['sdf_file_list'][index])
    features = fea_compiler.get_features(kernels, betas, taus, curv_types)
    X_test[test_index, :] = features
    y_test[test_index] = protein_test_dict['binding_energy_list'][index]
    # print(100*index/num_test_samples, 'percent done with test set')
    test_index += 1

data_dict = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'params': parameters}

# 2007 dataset
# data_filename = '/mnt/home/storeyd3/Documents/DG_Project/features/v2007_features/LL_v1' \
#   + '/data_dict_v1_' + sys.argv[1] 

# 2016 dataset
data_filename = '/mnt/home/storeyd3/Documents/DG_Project/features/v2016_features/LL_v1' \
  + '/data_dict_v1_' + sys.argv[1] 

with open(data_filename, 'wb') as file:
    pickle.dump(data_dict, file)