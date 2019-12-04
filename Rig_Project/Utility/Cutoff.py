import numpy as np
from scipy import spatial as sp

def get_cutoff_pairs(atom_list_1, atom_list_2, cutoff):
    pos_1 = np.zeros((len(atom_list_1), 3))
    pos_2 = np.zeros((len(atom_list_2), 3))

    for index in range(len(atom_list_1)):
        pos_1[index, :] = atom_list_1[index].pos
    for index in range(len(atom_list_2)):
        pos_2[index, :] = atom_list_2[index].pos

    dist_matrix = sp.distance.cdist(pos_1, pos_2, 'euclidean')

    atom_pair_list = []

    for i in range(len(atom_list_1)):
        for j in range(len(atom_list_2)):
            if dist_matrix[i,j] <= cutoff:
                atom_pair_list.append([atom_list_1[i], atom_list_2[j]])

    return atom_pair_list




