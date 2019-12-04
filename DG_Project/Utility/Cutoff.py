import numpy as np
from scipy import spatial as sp

def get_cutoff_list(atom, atom_list, cutoff):
    pos_1 = np.zeros((1,3))
    pos_2 = np.zeros((len(atom_list), 3))

    pos_1[0,:] = atom.pos

    for index in range(len(atom_list)):
        pos_2[index, :] = atom_list[index].pos


    dist_matrix = sp.distance.cdist(pos_1, pos_2, 'euclidean')

    output_list = []

    for i in range(len(atom_list)):
        if dist_matrix[0, i] > cutoff:
            output_list.append(atom_list[i])

    return atom_list

def get_binding_site(atom_list_1, atom_list_2, cutoff):
    pos_1 = np.zeros((len(atom_list_1), 3))
    pos_2 = np.zeros((len(atom_list_2), 3))

    for index in range(len(atom_list_1)):
        pos_1[index, :] = atom_list_1[index].pos
    for index in range(len(atom_list_2)):
        pos_2[index, :] = atom_list_2[index].pos

    dist_matrix = sp.distance.cdist(pos_1, pos_2, 'euclidean')

    output_protein_atom_list = []

    for i in range(len(atom_list_1)):
        if True in (dist_matrix[i,:] <= cutoff):
            output_protein_atom_list.append(atom_list_1[i])

    return output_protein_atom_list








