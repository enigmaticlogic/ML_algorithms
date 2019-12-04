import numpy as np
from Utility.Classes import atom
        
def read_pdb(pdb_file):

    '''
    Creates a list of atoms in the protein and assigns features
    obtained directly from the PDB to these atoms
    '''
    with open(pdb_file) as opened_pdb:
        list_pdb = opened_pdb.readlines()

    atom_list = []
    occupancy_condition = False
    for line in list_pdb:
        if 'ATOM' in line[0:7]:
            if float(line[54:60]) == 0.5:
                occupancy_condition = not occupancy_condition
            if float(line[54:60]) > 0.5 or occupancy_condition == True: 
                current_atom = atom()

                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                current_atom.pos = np.array([x,y,z])
                current_atom.heavy_type = line[13]
                atom_list.append(current_atom)

    return atom_list  