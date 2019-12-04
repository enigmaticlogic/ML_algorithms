import numpy as np
class atom:
    def __init__(self):
        '''
        pos = position data (x,y,z) (obtained directly from PDB)
        heavy_type = heavy element type of atom in one hot format, 5 choices (obtained directly from PDB)
        heavy element types are:
        ['C', 'N', 'O', 'S', 'H']
        atom_name = name of atom, ADD MORE DETAIL TO THIS LATER (obtained directly from PDB)
        Rval = R value, global feature of protein (obtained directly from PDB)
        B_factor = experimentally determined B Factor
        '''
        self.pos = np.zeros(3)
        self.heavy_type = 'none'
        self.atom_name = 'none'
        self.Rval = 0
        self.B_factor = 0
