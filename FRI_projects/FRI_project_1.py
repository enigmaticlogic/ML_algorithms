import numpy as np
import pandas as pd
import os
from scipy import spatial as sp
from scipy import stats
from sklearn.linear_model import LinearRegression as LinReg
%matplotlib inline 


def readPDB(PDBfile):

    '''
    Returns
    position data = [x,y,z]
    Bfactor_data [Bfactor]

    '''
    opened_PDB = open(PDBfile)
    list_PDB = opened_PDB.readlines()
    num_atoms = np.shape(list_PDB)[0]
    position_data = np.zeros((num_atoms, 3))
    Bfactor_data = np.zeros((num_atoms, 1))

    atom_index = 0
    for atom in list_PDB:
        if 'ATOM' in atom[0:7]: 
            x, y, z = float(atom[30:38]), float(atom[38:46]), float(atom[46:54])
            Bfactor = float(atom[60:66])
            position_data[atom_index,:] = np.array([x,y,z])
            Bfactor_data[atom_index,:] = Bfactor
            atom_index+=1
    
    return position_data, Bfactor_data

def compile_data(file_list, listdir):
    X_data = []
    y_data = []
    for file in file_list:
        X, y = readPDB(listdir+'/'+file)
        X_data.append(X)
        y_data.append(y)

    return X_data, y_data

def gaussian_kernel(dist, kappa, eta):
    return np.exp(-(dist/eta)**kappa)

def lorentz_kernel(dist, kappa, eta):
    return 1/(1+(dist/eta)**kappa)

def compute_flex_index(position_data, mode='11', kernel='gaussian', kappa=1, eta=1):
    dist_matrix = sp.distance.cdist(position_data, position_data, 'euclidean')
    mu = np.zeros(np.shape(position_data)[0])
    
    if mode == '11':
        if kernel == 'gaussian':
            for index in range(len(mu)):
                mu[index] = np.sum(gaussian_kernel(dist_matrix[index,:], kappa, eta))   
            mu_hat = mu/np.amax(mu)
            flex_index = 1/mu_hat
            return flex_index
        
        if kernel == 'lorentz':
            for index in range(len(mu)):
                mu[index] = np.sum(lorentz_kernel(dist_matrix[index,:], kappa, eta))   
            mu_hat = mu/np.amax(mu)
            flex_index = 1/mu_hat
            return flex_index
        
    if mode == '12':
        if kernel == 'gaussian':
            for index in range(len(mu)):
                mu[index] = np.sum(gaussian_kernel(dist_matrix[index,:], kappa, eta))   
            mu_hat = mu/np.amax(mu)
            flex_index = 1 - mu_hat
            return flex_index
        
        if kernel == 'lorentz':
            for index in range(len(mu)):
                mu[index] = np.sum(lorentz_kernel(dist_matrix[index,:], kappa, eta))   
            mu_hat = mu/np.amax(mu)
            flex_index = 1 - mu_hat
            return flex_index
        
    if mode == '21':
        if kernel == 'gaussian':
            for index in range(len(mu)):
                product_input = 1 - gaussian_kernel(dist_matrix[index,:], kappa, eta)
                product_input[index] = 1
                mu[index] = 1 - np.prod(product_input)
            mu_hat = mu/np.amax(mu)
            flex_index = 1/mu_hat
            return flex_index
        
        if kernel == 'lorentz':
            for index in range(len(mu)):
                product_input = 1 - lorentz_kernel(dist_matrix[index,:], kappa, eta)
                product_input[index] = 1
                mu[index] = 1 - np.prod(product_input)
            mu_hat = mu/np.amax(mu)
            flex_index = 1/mu_hat
            return flex_index
        
    if mode == '22':
        if kernel == 'gaussian':
            for index in range(len(mu)):
                product_input = 1 - gaussian_kernel(dist_matrix[index,:], kappa, eta)
                product_input[index] = 1
                mu[index] = 1 - np.prod(product_input)  
            mu_hat = mu/np.amax(mu)
            flex_index = 1 - mu_hat
            return flex_index
        
        if kernel == 'lorentz':
            for index in range(len(mu)):
                product_input = 1 - lorentz_kernel(dist_matrix[index,:], kappa, eta)
                product_input[index] = 1
                mu[index] = 1 - np.prod(product_input)
            mu_hat = mu/np.amax(mu)
            flex_index = 1 - mu_hat
            return flex_index
    
protein_listdir = 'C:/Users/Voltron Mini/Desktop/MLSummerStuff/Datasets/364_proteins/365'
protein_file_list = os.listdir(protein_listdir)
position_data, Bfactor_data = compile_data(protein_file_list, protein_listdir)
pCC_list = []

for index in range(len(position_data)):
    X = compute_flex_index(position_data[index], mode = '11', kernel = 'lorentz', kappa = 3, eta = 3)
    X = X.reshape(-1,1)
    y = Bfactor_data[index]

    reg = LinReg().fit(X,y)
    ypred = reg.predict(X)
    pCC = stats.pearsonr(ypred, y)[0]
    pCC_list.append(pCC)
    
pCC_array = np.array(pCC_list)
avg_pCC = np.average(pCC_array)
print(avg_pCC)