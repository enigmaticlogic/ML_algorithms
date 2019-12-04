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
                current_atom.heavy_type = line[13].strip()
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

                atom_list.append(current_atom)

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

    for single_atom in atom_list:
        single_atom.Rval = Rval

    return atom_list  