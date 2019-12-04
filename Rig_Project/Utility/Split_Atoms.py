def split_atoms(atom_list):
    C_atoms_list = []
    N_atoms_list = []
    O_atoms_list = []
    S_atoms_list = []
    H_atoms_list = []
    P_atoms_list = []
    F_atoms_list = []
    Cl_atoms_list = []
    Br_atoms_list = []
    I_atoms_list = []
    
    for atom in atom_list:
        if atom.heavy_type == 'C':
            C_atoms_list.append(atom)
        elif atom.heavy_type == 'N':
            N_atoms_list.append(atom)
        elif atom.heavy_type == 'O':
            O_atoms_list.append(atom)
        elif atom.heavy_type == 'S':
            S_atoms_list.append(atom)
        elif atom.heavy_type == 'H':
            H_atoms_list.append(atom)
        elif atom.heavy_type == 'P':
            P_atoms_list.append(atom)
        elif atom.heavy_type == 'F':
            F_atoms_list.append(atom)
        elif atom.heavy_type == 'Cl':
            Cl_atoms_list.append(atom)
        elif atom.heavy_type == 'Br':
            Br_atoms_list.append(atom)
        elif atom.heavy_type == 'I':
            I_atoms_list.append(atom)

    split_atoms = {'C': C_atoms_list, 'N': N_atoms_list, 'O': O_atoms_list, 'S': S_atoms_list, 'H': H_atoms_list,\
    'P': P_atoms_list, 'F': F_atoms_list, 'Cl': Cl_atoms_list, 'Br': Br_atoms_list, 'I': I_atoms_list}

    return split_atoms