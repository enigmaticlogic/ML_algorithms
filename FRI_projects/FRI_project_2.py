import numpy as np
import pandas as pd
import os
from scipy import spatial as sp
from scipy import stats
from sklearn.linear_model import LinearRegression as LinReg
%matplotlib inline 

#-----------------------FUNCIONS---------------------#
def readPDB(PDBfile):

    '''
    Returns
    atom_data = ['type', x, y, z, Bfactor]

    '''
    opened_PDB = open(PDBfile)
    list_PDB = opened_PDB.readlines()
    num_atoms = 0
    for atom in list_PDB:
        if 'ATOM' in atom[0:7]:
            num_atoms += 1
    protein_data = np.zeros((num_atoms, 5))

    atom_index = 0
    for atom in list_PDB:
        if 'ATOM' in atom[0:7]: 
            if 'CA' in atom[12:16]:
                atom_type = 1
            elif 'C' in atom[76:78]:
                atom_type = 2
            elif 'N' in atom[76:78]:
                atom_type = 3
            elif 'O' in atom[76:78]:
                atom_type = 4
            elif 'S' in atom[76:78]:
                atom_type = 5   
            else:
                atom_type = 0
            x, y, z = float(atom[30:38]), float(atom[38:46]), float(atom[46:54])
            Bfactor = float(atom[60:66])
            protein_data[atom_index,:] = np.array([atom_type, x, y, z, Bfactor])
            atom_index+=1
    
    return protein_data

def split_atom_data(atom_data):
    CA_pos_data = atom_data[np.where(atom_data == 1)[0], 1:5] #Calpha, also includes B factor
    CNA_pos_data = atom_data[np.where(atom_data == 2)[0], 1:5] #Non Calpha Carbon, also includes B factor
    C_pos_data = atom_data[np.where(np.logical_or(atom_data == 1, atom_data == 2))[0], 1:5]
    N_pos_data = atom_data[np.where(atom_data == 3)[0], 1:5]
    O_pos_data = atom_data[np.where(atom_data == 4)[0], 1:5]
    S_pos_data = atom_data[np.where(atom_data == 5)[0], 1:5]
    return C_pos_data, N_pos_data, O_pos_data, S_pos_data, CA_pos_data, CNA_pos_data

def compile_data(file_list, listdir):
    atom_data = []
    for file in file_list:
        one_protein = readPDB(listdir+'/'+file)
        atom_data.append(one_protein)

    return atom_data

def compile_S_data(file_list, listdir):
    atom_data = []
    for file in file_list:
        one_protein = readPDB(listdir+'/'+file)
        if 5 in one_protein[:,0]:
            unique, counts = np.unique(one_protein[:,0], return_counts=True)
            counts_dict = dict(zip(unique, counts))
            if counts_dict[5] > 3:
                atom_data.append(one_protein)
    
    return atom_data

def gaussian_ker(dist, kappa, eta):
    return np.exp(-(dist/eta)**kappa)

def lorentz_ker(dist, kappa, eta):
    return 1/(1+(dist/eta)**kappa)

def compute_flex_index(atom_data, mode='O', kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'], \
                       kappa=[1,1,1], eta=[1,1,1]):
    C_data, N_data, O_data, S_data, CA_data, CNA_data = split_atom_data(atom_data)
    
    if mode == 'CA':
        C_dist_matrix = sp.distance.cdist(CA_data[:, 0:3], C_data[:, 0:3], 'euclidean')
        N_dist_matrix = sp.distance.cdist(CA_data[:, 0:3], N_data[:, 0:3], 'euclidean')
        O_dist_matrix = sp.distance.cdist(CA_data[:, 0:3], O_data[:, 0:3], 'euclidean')
        num_atoms = np.shape(CA_data)[0]
        mu = np.zeros((num_atoms, 9))
        
    if mode == 'CNA':
        C_dist_matrix = sp.distance.cdist(CNA_data[:, 0:3], C_data[:, 0:3], 'euclidean')
        N_dist_matrix = sp.distance.cdist(CNA_data[:, 0:3], N_data[:, 0:3], 'euclidean')
        O_dist_matrix = sp.distance.cdist(CNA_data[:, 0:3], O_data[:, 0:3], 'euclidean')
        num_atoms = np.shape(CNA_data)[0]
        mu = np.zeros((num_atoms, 9))
        
    if mode == 'N':
        C_dist_matrix = sp.distance.cdist(N_data[:, 0:3], C_data[:, 0:3], 'euclidean')
        N_dist_matrix = sp.distance.cdist(N_data[:, 0:3], N_data[:, 0:3], 'euclidean')
        O_dist_matrix = sp.distance.cdist(N_data[:, 0:3], O_data[:, 0:3], 'euclidean')
        num_atoms = np.shape(N_data)[0]
        mu = np.zeros((num_atoms, 9))
        
    if mode == 'O':
        C_dist_matrix = sp.distance.cdist(O_data[:, 0:3], C_data[:, 0:3], 'euclidean')
        N_dist_matrix = sp.distance.cdist(O_data[:, 0:3], N_data[:, 0:3], 'euclidean')
        O_dist_matrix = sp.distance.cdist(O_data[:, 0:3], O_data[:, 0:3], 'euclidean')
        num_atoms = np.shape(O_data)[0]
        mu = np.zeros((num_atoms, 9))
        
    if mode == 'S':
        C_dist_matrix = sp.distance.cdist(S_data[:, 0:3], C_data[:, 0:3], 'euclidean')
        N_dist_matrix = sp.distance.cdist(S_data[:, 0:3], N_data[:, 0:3], 'euclidean')
        O_dist_matrix = sp.distance.cdist(S_data[:, 0:3], O_data[:, 0:3], 'euclidean')
        num_atoms = np.shape(S_data)[0]
        mu = np.zeros((num_atoms, 9))
    
    for index in range(num_atoms):
        mu[index, 0] = np.sum(globals()[kernel[0]](C_dist_matrix[index,:], kappa[0], eta[0]))
        mu[index, 1] = np.sum(globals()[kernel[0]](N_dist_matrix[index,:], kappa[0], eta[0]))
        mu[index, 2] = np.sum(globals()[kernel[0]](O_dist_matrix[index,:], kappa[0], eta[0]))
        mu[index, 3] = np.sum(globals()[kernel[1]](C_dist_matrix[index,:], kappa[1], eta[1]))
        mu[index, 4] = np.sum(globals()[kernel[1]](N_dist_matrix[index,:], kappa[1], eta[1]))
        mu[index, 5] = np.sum(globals()[kernel[1]](O_dist_matrix[index,:], kappa[1], eta[1]))
        mu[index, 6] = np.sum(globals()[kernel[2]](C_dist_matrix[index,:], kappa[2], eta[2]))
        mu[index, 7] = np.sum(globals()[kernel[2]](N_dist_matrix[index,:], kappa[2], eta[2]))
        mu[index, 8] = np.sum(globals()[kernel[2]](O_dist_matrix[index,:], kappa[2], eta[2]))
    flex_index = 1/mu
    return flex_index

#---------------------GATHERING RESULTS---------------------#
protein_listdir = 'C:/Users/Voltron Mini/Desktop/MLSummerStuff/Datasets/park_small'
protein_file_list = os.listdir(protein_listdir)
atom_data = compile_data(protein_file_list, protein_listdir)

small_pCC_list = []
for index in range(len(atom_data)):
    C_data, N_data, O_data, S_data, CA_data, CNA_data = split_atom_data(atom_data[index])
    
    mode_choice = 'CA' #Choose mode here
    X = compute_flex_index(atom_data[index], mode=mode_choice, kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'], \
                           kappa=[1,3,1], eta=[16,2,31])
    y = CA_data[:,3]
    reg = LinReg().fit(X,y)
    ypred = reg.predict(X)
    pCC = stats.pearsonr(ypred, y)[0]
    small_pCC_list.append(pCC)

small_pCC_array = np.array(small_pCC_list)
small_avg_pCC = np.average(small_pCC_array)
print('Small C alpha carbon average Pearson correlation coefficient: ', small_avg_pCC)

protein_listdir = 'C:/Users/Voltron Mini/Desktop/MLSummerStuff/Datasets/park_medium'
protein_file_list = os.listdir(protein_listdir)
atom_data = compile_data(protein_file_list, protein_listdir)

medium_pCC_list = []
for index in range(len(atom_data)):
    C_data, N_data, O_data, S_data, CA_data, CNA_data = split_atom_data(atom_data[index])
    
    mode_choice = 'CA' #Choose mode here
    X = compute_flex_index(atom_data[index], mode=mode_choice, kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'], \
                           kappa=[1,3,1], eta=[16,2,31])
    y = CA_data[:,3]
    reg = LinReg().fit(X,y)
    ypred = reg.predict(X)
    pCC = stats.pearsonr(ypred, y)[0]
    medium_pCC_list.append(pCC)

medium_pCC_array = np.array(medium_pCC_list)
medium_avg_pCC = np.average(medium_pCC_array)
print('Medium C alpha carbon average Pearson correlation coefficient: ', medium_avg_pCC)

protein_listdir = 'C:/Users/Voltron Mini/Desktop/MLSummerStuff/Datasets/park_large'
protein_file_list = os.listdir(protein_listdir)
atom_data = compile_data(protein_file_list, protein_listdir)

large_pCC_list = []
for index in range(len(atom_data)):
    C_data, N_data, O_data, S_data, CA_data, CNA_data = split_atom_data(atom_data[index])
    
    mode_choice = 'CA' #Choose mode here
    X = compute_flex_index(atom_data[index], mode=mode_choice, kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'], \
                           kappa=[1,3,1], eta=[16,2,31])
    y = CA_data[:,3]
    reg = LinReg().fit(X,y)
    ypred = reg.predict(X)
    pCC = stats.pearsonr(ypred, y)[0]
    large_pCC_list.append(pCC)

large_pCC_array = np.array(large_pCC_list)
large_avg_pCC = np.average(large_pCC_array)
print('Large C alpha carbon average Pearson correlation coefficient: ', large_avg_pCC)

protein_listdir = 'C:/Users/Voltron Mini/Desktop/MLSummerStuff/Datasets/park_full'
protein_file_list = os.listdir(protein_listdir)
atom_data = compile_data(protein_file_list, protein_listdir)

superset_pCC_list = []
for index in range(len(atom_data)):
    C_data, N_data, O_data, S_data, CA_data, CNA_data = split_atom_data(atom_data[index])
    
    mode_choice = 'CA' #Choose mode here
    X = compute_flex_index(atom_data[index], mode=mode_choice, kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'], \
                           kappa=[1,3,1], eta=[16,2,31])
    y = CA_data[:,3]
    reg = LinReg().fit(X,y)
    ypred = reg.predict(X)
    pCC = stats.pearsonr(ypred, y)[0]
    superset_pCC_list.append(pCC)

superset_pCC_array = np.array(superset_pCC_list)
superset_avg_pCC = np.average(superset_pCC_array)
print('Superset C alpha carbon average Pearson correlation coefficient: ', superset_avg_pCC)

protein_listdir = 'C:/Users/Voltron Mini/Desktop/MLSummerStuff/Datasets/364_proteins_full'
protein_file_list = os.listdir(protein_listdir)
atom_data = compile_S_data(protein_file_list, protein_listdir)

S_pCC_list = []
for index in range(len(atom_data)):
    C_data, N_data, O_data, S_data, CA_data, CNA_data = split_atom_data(atom_data[index])
    
    mode_choice = 'S' #Choose mode here
    X = compute_flex_index(atom_data[index], mode=mode_choice, kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'], \
                           kappa=[1,3,1], eta=[16,2,31])
    y = S_data[:,3]
    reg = LinReg().fit(X,y)
    ypred = reg.predict(X)
    pCC = stats.pearsonr(ypred, y)[0]
    S_pCC_list.append(pCC)
    
S_pCC_array = np.array(S_pCC_list)
S_avg_pCC = np.average(S_pCC_array)
print('Sulphur average Pearson correlation coefficient: ', S_avg_pCC)

protein_listdir = 'C:/Users/Voltron Mini/Desktop/MLSummerStuff/Datasets/364_proteins_full'
protein_file_list = os.listdir(protein_listdir)
atom_data = compile_data(protein_file_list, protein_listdir)

CA_pCC_list, CNA_pCC_list, N_pCC_list, O_pCC_list = [], [], [], []
for index in range(len(atom_data)):
    C_data, N_data, O_data, S_data, CA_data, CNA_data = split_atom_data(atom_data[index])
    
    mode_choice = 'CA' #Choose mode here
    X = compute_flex_index(atom_data[index], mode=mode_choice, kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'], \
                           kappa=[1,3,1], eta=[16,2,31])
    y = CA_data[:,3]
    reg = LinReg().fit(X,y)
    ypred = reg.predict(X)
    pCC = stats.pearsonr(ypred, y)[0]
    CA_pCC_list.append(pCC)
    
    mode_choice = 'CNA' #Choose mode here
    X = compute_flex_index(atom_data[index], mode=mode_choice, kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'], \
                           kappa=[1,3,1], eta=[16,2,31])
    y = CNA_data[:,3]
    reg = LinReg().fit(X,y)
    ypred = reg.predict(X)
    pCC = stats.pearsonr(ypred, y)[0]
    CNA_pCC_list.append(pCC)
    
    mode_choice = 'N' #Choose mode here
    X = compute_flex_index(atom_data[index], mode=mode_choice, kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'], \
                           kappa=[1,3,1], eta=[16,2,31])
    y = N_data[:,3]
    reg = LinReg().fit(X,y)
    ypred = reg.predict(X)
    pCC = stats.pearsonr(ypred, y)[0]
    N_pCC_list.append(pCC)
    
    mode_choice = 'O' #Choose mode here
    X = compute_flex_index(atom_data[index], mode=mode_choice, kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'], \
                           kappa=[1,3,1], eta=[16,2,31])
    y = O_data[:,3]
    reg = LinReg().fit(X,y)
    ypred = reg.predict(X)
    pCC = stats.pearsonr(ypred, y)[0]
    O_pCC_list.append(pCC)
    
CA_pCC_array = np.array(CA_pCC_list)
CA_avg_pCC = np.average(CA_pCC_array)
print('C alpha average Pearson correlation coefficient: ', CA_avg_pCC)

CNA_pCC_array = np.array(CNA_pCC_list)
CNA_avg_pCC = np.average(CNA_pCC_array)
print('Non C alpha carbon average Pearson correlation coefficient: ', CNA_avg_pCC)

N_pCC_array = np.array(N_pCC_list)
N_avg_pCC = np.average(N_pCC_array)
print('Nitrogen average Pearson correlation coefficient: ', N_avg_pCC)

O_pCC_array = np.array(O_pCC_list)
O_avg_pCC = np.average(O_pCC_array)
print('Oxygen average Pearson correlation coefficient: ', O_avg_pCC)
