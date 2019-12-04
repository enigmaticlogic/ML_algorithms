import numpy as np
class atom:
    def __init__(self):
        '''
        pos = position data (x,y,z) (obtained directly from PDB)
        heavy_type = heavy element type of atom in one hot format, 5 choices (obtained directly from PDB)
        heavy element types are:
        ['C', 'N', 'O', 'S', 'H', 'P', 'F', 'Cl', 'Br', 'I']
        '''
        self.pos = np.zeros(3)
        self.heavy_type = 'none'
