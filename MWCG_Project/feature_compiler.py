import numpy as np
import pickle
import os
import sys
from scipy import spatial as sp

#----------------------------------Classes-------------------------#

class atom:
    def __init__(self):
        '''
        pos = position data (x,y,z) (obtained directly from PDB)
        res_number = which residue the atom belongs to
        amino_type = amino type of atom in one hot format, 20 choices (obtained directly from PDB)
        amino types are:
        ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
        'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        heavy_type = heavy element type of atom in one hot format, 5 choices (obtained directly from PDB)
        heavy element types are:
        ['C', 'N', 'O', 'S', 'H']
        atom_name = name of atom, ADD MORE DETAIL TO THIS LATER (obtained directly from PDB)
        rig = rigidity obtained from MWCG, 9 values
        flex = flexibility obtained from MWCG, used for CNN
        packing_density = packing density for atom (small, med, large)
        structure = STRIDE secondary structure data for atom
        structure types are:
        [alpha helix, 3-10 helix, PI-helix, extended confromation/strand, isolated bridge, turn, coil]
        given by ['H', 'G', 'I', 'E', 'B', 'T', 'C']
        phi = phi angle obtained directly from STRIDE file
        psi = psi angle obtained directly from STRIDE file
        solv_area = obtained directly from STRIDE file
        Rval = R value, global feature of protein (obtained directly from PDB)
        res = resolution, global feature of protein (obtained directly from PDB)
        num_heavy_atoms = number of heavy atoms in one hot format via cutoffs, global feature of protein
        B_factor = experimentally determined B Factor
        '''
        self.pos = np.zeros(3)
        self.res_number = None
        self.amino_type = np.zeros(20)
        self.heavy_type = np.zeros(5)
        self.atom_name = 'none'
        self.rig = np.zeros(9)
        self.flex = np.zeros((8,30))
        self.packing_density = np.zeros(3)
        self.structure = np.zeros(7)
        self.phi = 0
        self.psi = 0
        self.solv_area = 0
        self.Rval = 0
        self.res = 0
        self.num_heavy_atoms = np.zeros(10)
        self.B_factor = 0
        self.CNN_image = np.zeros((8,30,3))
        
class residue:
    def __init__(self):
        '''
        res_number = residue ID used to tell which atoms should be given which feature values
        structure = Structure type of the residue, 7 choices (obtained from STRIDE file)
        structure types: ['H', 'G', 'I', 'E', 'B', 'T', 'C']
        phi = angle 
        psi = angle 
        solv_area = area 
        '''
        self.res_number = None
        self.structure = None
        self.phi = None
        self.psi = None
        self.solv_area = None

class feature_compiler:
    def __init__(self, file, STRIDE_file, protein_ID):
        '''
        Class for computing the features for a single protein
        file = protein PDB file
        STRIDE_file = STRIDE secondary feature file for the protein
        '''
        self.file = file
        self.STRIDE_file = STRIDE_file
        self.protein_ID = protein_ID
        
    def readPDB(self, PDBfile):

        '''
        Creates a list of atoms in the protein and assigns features
        obtained directly from the PDB to these atoms
        '''
        with open(PDBfile) as opened_PDB:
            list_PDB = opened_PDB.readlines()

        atom_index = 0
        self.atom_list = []
        amino_types = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', \
                              'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        heavy_types = ['C', 'N', 'O', 'S', 'H'] 
        occupancy_condition = False
        for line in list_PDB:
            if 'ATOM' in line[0:7]:
                if float(line[54:60]) == 0.5:
                    occupancy_condition = not occupancy_condition
                    print(occupancy_condition)
                if float(line[54:60]) > 0.5 or occupancy_condition == True: 
                    current_atom = atom()

                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    current_atom.pos = np.array([x,y,z])
                    
                    current_atom.res_number = line[22:26].strip() + line[21:22].strip() # Chain identifier
                    current_atom_res_name = line[17:20].strip()
                    current_atom_amino_index = amino_types.index(current_atom_res_name)
                    current_atom.amino_type[current_atom_amino_index] = 1

                    current_atom_heavy_str = line[77:78].strip()
                    current_atom_heavy_index = heavy_types.index(current_atom_heavy_str)
                    current_atom.heavy_type[current_atom_heavy_index] = 1
                    current_atom.B_factor = float(line[60:66])


                    if 'CA' in line[12:16]:
                        current_atom.atom_name = 'CA'
                    elif 'C' in line[76:78]:
                        current_atom.atom_name = 'C'
                    elif 'N' in line[76:78]:
                        current_atom.atom_name = 'N'
                    elif 'O' in line[76:78]:
                        current_atom.atom_name = 'O'
                    elif 'S' in line[76:78]:
                        current_atom.atom_name = 'S' 
                    elif 'H' in line[76:78]:
                        current_atom.atom_name = 'H'

                    if current_atom.atom_name != 'H':    
                        self.atom_list.append(current_atom)
                        atom_index+=1

            if 'RESOLUTION RANGE HIGH (ANGSTROMS)' in line:
                res = float(line[49:53])

            if 'R VALUE            (WORKING SET) :' in line:
                Rval_1 = line[48:53].strip()            
            if 'FREE R VALUE                     :' in line:
                Rval_2 = line[48:53].strip()
            if 'R VALUE          (WORKING SET, NO CUTOFF) :' in line:
                Rval_3 = line[57:63].strip()
            if 'R VALUE     (WORKING + TEST SET) :' in line:
                Rval_4 = line[48:53].strip()
            if 'FREE R VALUE                  (NO CUTOFF) :' in line:
                Rval_5 = line[57:63].strip()
            if 'R VALUE          (WORKING SET, F>4SIG(F)) :' in line:
                Rval_6 = line[57:63].strip()

        num_heavy_atoms_ohe = np.zeros(10)
        if atom_index < 500:
            num_heavy_atoms_ohe[0] = 1
        elif atom_index < 750:
            num_heavy_atoms_ohe[1] = 1
        elif atom_index < 1000:
            num_heavy_atoms_ohe[2] = 1
        elif atom_index < 1500:
            num_heavy_atoms_ohe[3] = 1
        elif atom_index < 2000:
            num_heavy_atoms_ohe[4] = 1
        elif atom_index < 2500:
            num_heavy_atoms_ohe[5] = 1
        elif atom_index < 3000:
            num_heavy_atoms_ohe[6] = 1
        elif atom_index < 4000:
            num_heavy_atoms_ohe[7] = 1
        elif atom_index < 5000:
            num_heavy_atoms_ohe[8] = 1
        elif atom_index < 30000:
            num_heavy_atoms_ohe[9] = 1

        try:
            Rval = float(Rval_1)
        except:
            try:
                Rval = float(Rval_2)
            except:
                try:
                    Rval = float(Rval_3)
                except:
                    try:
                        Rval = float(Rval_4)
                    except:
                        try:
                            Rval = float(Rval_5)
                        except:
                            try:
                                Rval = float(Rval_6)
                            except:
                                Rval = 0

        for single_atom in self.atom_list:
            single_atom.res = res
            single_atom.Rval = Rval
            single_atom.num_heavy_atoms = num_heavy_atoms_ohe

        print('Rval is: ', Rval)
            
    def readSTRIDE(self, STRIDE_file):
        '''
        Uses the STRIDE file to compute secondary features for each atom based
        on which residue the atom is contained in.
        All atoms in the same residue are given the same secondary feature values.
        '''
        with open(STRIDE_file) as opened_STRIDE:
            list_STRIDE = opened_STRIDE.readlines()
        self.residue_list = []
        
        for line in list_STRIDE:
            if 'ASG' in line[0:7]:
                current_residue = residue()
                current_residue.res_number = line[11:15].strip() + line[9:10].strip() # Chain identifier
                current_residue.structure = line[24]
                current_residue.phi = float(line[42:49])
                current_residue.psi = float(line[52:59])
                current_residue.solv_area = float(line[64:69])
                self.residue_list.append(current_residue)
            
    def compile_STRIDE_data(self):
        counter = 0
        for single_atom in self.atom_list:
            for single_residue in self.residue_list:
                if single_atom.res_number == single_residue.res_number:
                    counter += 1                    
                    if single_residue.structure == 'H':
                        single_atom.structure[0] = 1
                    elif single_residue.structure == 'G':
                        single_atom.structure[1] = 1
                    elif single_residue.structure == 'I':
                        single_atom.structure[2] = 1
                    elif single_residue.structure == 'E':
                        single_atom.structure[3] = 1
                    elif single_residue.structure == 'B':
                        single_atom.structure[4] = 1
                    elif single_residue.structure == 'T':
                        single_atom.structure[5] = 1
                    elif single_residue.structure == 'C':
                        single_atom.structure[6] = 1
                    single_atom.phi = single_residue.phi
                    single_atom.psi = single_residue.psi
                    single_atom.solv_area = single_residue.solv_area
                        
    def split_atoms_by_element_type(self):
        '''
        Creates sublists of the atom list containing atoms 
        of a single heavy atom type for computational purposes.
        '''
        self.C_atoms_list = []
        self.N_atoms_list = []
        self.O_atoms_list = []
        for single_atom in self.atom_list:
            if single_atom.atom_name == 'C' or single_atom.atom_name == 'CA':
                self.C_atoms_list.append(single_atom)
            elif single_atom.atom_name == 'N':
                self.N_atoms_list.append(single_atom)
            elif single_atom.atom_name == 'O':
                self.O_atoms_list.append(single_atom)
                
    def compute_mu(self, kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'],\
            kappa=[1,1,1], eta=[1,1,1]):
        '''
        Computes mu via the MWCG method outlined in Bramer's paper
        '''
        # Array of distance data for each split
        self.total_dd = np.zeros((len(self.atom_list), 3))
        self.C_dd = np.zeros((len(self.C_atoms_list), 3))
        self.N_dd = np.zeros((len(self.N_atoms_list), 3))
        self.O_dd = np.zeros((len(self.O_atoms_list), 3))
        
        for index in range(len(self.atom_list)):
            self.total_dd[index, :] = self.atom_list[index].pos
        for index in range(len(self.C_atoms_list)):
            self.C_dd[index, :] = self.C_atoms_list[index].pos
        for index in range(len(self.N_atoms_list)):
            self.N_dd[index, :] = self.N_atoms_list[index].pos
        for index in range(len(self.O_atoms_list)):
            self.O_dd[index, :] = self.O_atoms_list[index].pos
            
        C_dist_matrix = sp.distance.cdist(self.total_dd, self.C_dd, 'euclidean')
        N_dist_matrix = sp.distance.cdist(self.total_dd, self.N_dd, 'euclidean')
        O_dist_matrix = sp.distance.cdist(self.total_dd, self.O_dd, 'euclidean')
        self.total_dist_matrix = sp.distance.cdist(self.total_dd, self.total_dd, 'euclidean')

        num_atoms = len(self.atom_list)
        mu = np.zeros((num_atoms, 9))
        
        # 9 mu values are computed for each atom, 3 scales and 3 distance pairings
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
            self.atom_list[index].rig = mu[index, :]

        mu_2 = np.zeros((num_atoms, 8, 30, 3))
        MWCG_image_kappas = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 9, 10, 11]
        MWCG_image_etas = [1, 2, 3, 4, 5, 10, 15, 20]
            
        for index in range(num_atoms):
            for eta_index in range(len(MWCG_image_etas)):
                for kappa_index in range(len(MWCG_image_kappas)):
                    mu_2[index, eta_index, kappa_index, 0] = \
                      np.sum(gaussian_ker(C_dist_matrix[index,:],\
                        MWCG_image_kappas[kappa_index], MWCG_image_etas[eta_index]))
                    mu_2[index, eta_index, kappa_index, 1] = \
                      np.sum(gaussian_ker(N_dist_matrix[index,:],\
                        MWCG_image_kappas[kappa_index], MWCG_image_etas[eta_index]))
                    mu_2[index, eta_index, kappa_index, 2] = \
                      np.sum(gaussian_ker(O_dist_matrix[index,:],\
                        MWCG_image_kappas[kappa_index], MWCG_image_etas[eta_index]))
                    mu_2[index, eta_index, kappa_index + 15, 0] = \
                      np.sum(lorentz_ker(C_dist_matrix[index,:],\
                        MWCG_image_kappas[kappa_index], MWCG_image_etas[eta_index]))
                    mu_2[index, eta_index, kappa_index + 15, 1] = \
                      np.sum(lorentz_ker(N_dist_matrix[index,:],\
                        MWCG_image_kappas[kappa_index], MWCG_image_etas[eta_index]))
                    mu_2[index, eta_index, kappa_index + 15, 2] = \
                      np.sum(lorentz_ker(O_dist_matrix[index,:],\
                        MWCG_image_kappas[kappa_index], MWCG_image_etas[eta_index]))
            self.atom_list[index].CNN_image = mu_2[index, :] 
            
    def compute_packing_density(self, cutoffs = [3, 5]):
        '''
        Computes the packing density for each atom.
        Could be improved via cell list.
        '''
        
        for index in range(len(self.atom_list)):
            N_1 = sum(i < cutoffs[0] for i in self.total_dist_matrix[index, :])
            N_2 = sum(i >= cutoffs[0] and i < cutoffs[1] for i in self.total_dist_matrix[index, :])
            N_3 = sum(i >= cutoffs[1] for i in self.total_dist_matrix[index, :])
            self.atom_list[index].packing_density = np.array([N_1, N_2, N_3])/len(self.atom_list)
            
    def compute_all_features(self, kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'],\
        kappa=[1,1,1], eta=[1,1,1], cutoffs = [3,5]):
        '''
        Computes all features for a single protein and returns a dictionary with
        keys 'protein_ID' = protein name, 'features' = array of features (num_atoms x num_features).
        '''
        self.readPDB(self.file)
        self.readSTRIDE(self.STRIDE_file)
        self.compile_STRIDE_data()
        self.split_atoms_by_element_type()
        self.compute_mu(kernel, kappa, eta) # Can change kernel types and constants kappa, eta here
        self.compute_packing_density(cutoffs)
        combined_feature_array = np.zeros((len(self.atom_list), 61)) # 59 features plus b factor and CA bool tag
        CNN_input = np.zeros((len(self.atom_list), 8, 30, 3)) # MWCG inputs for the CNN
        for index in range(len(self.atom_list)):
            combined_feature_array[index, 0:20] = self.atom_list[index].amino_type
            combined_feature_array[index, 20:25] = self.atom_list[index].heavy_type
            combined_feature_array[index, 25:34] = self.atom_list[index].rig
            combined_feature_array[index, 34:37] = self.atom_list[index].packing_density
            combined_feature_array[index, 37:38] = self.atom_list[index].Rval
            combined_feature_array[index, 38:39] = self.atom_list[index].res
            combined_feature_array[index, 39:46] = self.atom_list[index].structure
            combined_feature_array[index, 46:47] = self.atom_list[index].phi
            combined_feature_array[index, 47:48] = self.atom_list[index].psi
            combined_feature_array[index, 48:49] = self.atom_list[index].solv_area
            combined_feature_array[index, 49:59] = self.atom_list[index].num_heavy_atoms
            combined_feature_array[index, 59] = self.atom_list[index].B_factor
            if self.atom_list[index].atom_name == 'CA':
                combined_feature_array[index, 60] = 1
            else:
                combined_feature_array[index, 60] = 0

            CNN_input[index, :] = self.atom_list[index].CNN_image

        combined_feature_dict = {'protein_ID': self.protein_ID, 'features': combined_feature_array,\
            'CNN_input': CNN_input}
        return combined_feature_dict
        
        
#------------------------------Functions-------------------------------------#

def gaussian_ker(dist, kappa, eta):
    return np.exp(-(dist/eta)**kappa)

def lorentz_ker(dist, kappa, eta):
    return 1/(1+(dist/eta)**kappa)
        
def compile_data(file_list, STRIDE_file_list, protein_ID_list):
    '''
    Computes feature values for all protein PDB and STRIDE files in a list.
    Returns a list of arrays of feature values (num_atoms x num_features).
    '''
    all_proteins_feature_list = []
    count = 0
    for index in range(len(file_list)):
        fea_compile = feature_compiler(file_list[index], STRIDE_file_list[index], protein_ID_list[index])
        features_one_protein = fea_compile.compute_all_features\
        (kernel=['lorentz_ker', 'lorentz_ker', 'gaussian_ker'], kappa=[3,1,1], eta=[16,2,31], cutoffs = [3,5])
        all_proteins_feature_list.append(features_one_protein)
        count += 1
        print(count)

    return all_proteins_feature_list

#------------------------------Execution-------------------------------------#

protein_dir = '/mnt/home/storeyd3/Documents/Datasets/park_medium_trim' # Dir to get PDB's from
protein_ID_list = sorted(os.listdir(protein_dir))
protein_file_list = []
for file in sorted(os.listdir(protein_dir)):
    protein_file_list.append(protein_dir+'/'+file)

STRIDE_dir = '/mnt/home/storeyd3/Documents/Datasets/STRIDE_data/park_medium_trim' # Dir to get STRIDE's from
STRIDE_file_list = []
for file in sorted(os.listdir(STRIDE_dir)):
    STRIDE_file_list.append(STRIDE_dir+'/'+file)

start_index = int(sys.argv[1]) - 1

protein_file_list = protein_file_list[start_index:start_index+1]
STRIDE_file_list = STRIDE_file_list[start_index:start_index+1]
protein_ID_list = protein_ID_list[start_index:start_index+1]

all_proteins_feature_dict = compile_data(protein_file_list, STRIDE_file_list, protein_ID_list)
print(all_proteins_feature_dict[0]['features'].shape)
# Dir to save pickled outputs to
fea_filename = '/mnt/home/storeyd3/Documents/Jobs/features/medium/'+'protein_fea'+sys.argv[1]

with open(fea_filename, 'wb') as file:
    pickle.dump(all_proteins_feature_dict, file)