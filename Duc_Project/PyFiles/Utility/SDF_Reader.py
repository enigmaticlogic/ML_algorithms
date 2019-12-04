import numpy as np
from Utility.Classes import atom
        
def read_sdf(sdf_file):

    '''
    Creates a list of atoms in the ligand and assigns features
    obtained directly from the SDF to these atoms
    '''
    with open(sdf_file) as opened_sdf:
        list_sdf = opened_sdf.readlines()

    atom_list = []
    heavy_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'] 
    for line in list_sdf:
        split_line = line.split()
        if len(split_line) == 9:
            current_atom = atom()
            x, y, z = float(split_line[0]), float(split_line[1]), float(split_line[2])
            current_atom.pos = np.array([x,y,z])
            current_atom.heavy_type = split_line[3]  
            
            atom_list.append(current_atom)

    return atom_list