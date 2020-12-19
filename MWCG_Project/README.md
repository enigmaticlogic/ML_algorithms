<details>
  <summary>Project Overview</summary>
  
  ## Intro
  Hello! The purpose of this project was to reproduce the results of *Blind Prediction of Protein B-factor and Flexibility* by David Bramer and Guo-Wei Wei. A PDF of the paper is included in the repo. The code used to reproduce the results is meant to run on MSU's high performance computing cluster, so instead of showing you how to run it yourself I will walk you through an overview of the code and methods used.
  
</details>

<details>
  <summary>Extracting the Data</summary>
  
  ## PDB Files and Features Used
  The training and test data for this project comes from the protein databank in the form of PDB files. These are plaintext files which contain information obtained through xray crystallography about proteins. There are global features, which apply to all atoms within a protein, and local features which are dependent on the atom. Examples of global features used are the resolution (gives a notion of the quality of the protein model) and the number of heavy atoms (gives a notion of the size of the protein). The construction of local features which represent local rigidity of the structure is where a lot of the work of the paper lies. The idea of Multi Weighted Colored Graphs (MWCGs) are used to generate a rigidity index for an atom based on the position and element type pair interactions, which are included in the PDB files. Also included are 12 secondary features which are generated from a program called STRIDE, which categorize atoms as belonging to sub structures such as helixes or coils. The residue number of an atom obtained from the PDB file determines which atoms should share secondary features.
  
  An atom class is included in feature_compiler.py, which contains the following class variables:
  
  ```
  pos = position data (x,y,z) (obtained directly from PDB)
  res_number = which residue the atom belongs to
  amino_type = amino type of atom in one hot format, 20 choices (obtained directly from PDB)
  amino types are:
  ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
  'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
  heavy_type = heavy element type of atom in one hot format, 5 choices (obtained directly from PDB)
  heavy element types are:
  ['C', 'N', 'O', 'S', 'H']
  atom_name = name of atom (obtained directly from PDB)
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
  ```
  
  Some of these values are pulled directly from the PDB file, but many of them are calculated using the element type and position information from the PDB file. There is also a value for atoms called the occupancy condition, which determines the probability that a specific atom will occupy a certain position. For a single position, the occupancy condition of all atoms at that position should sum to 1. For this reason, we only consider atoms with occupancy greater than .5, or when two atoms have an occupancy condition of 0.5, we only consider one of them. This is handled in the below lines:
  
  ```
  if float(line[54:60]) == 0.5:
      occupancy_condition = not occupancy_condition
      print(occupancy_condition)
  if float(line[54:60]) > 0.5 or occupancy_condition == True: 
      current_atom = atom()
  ```
  
  The readPDB method performs this check for each atom, extracts all needed features, and adds the atoms to a list. This extraction required a great amount of familiarty with the data set to deal with situations such as the occupancy condition scenario described above. To construct the global feature from the nubmer of heavy atoms, one hot encoding was used with defined cutoffs representing various size categories. After extracting the neccessary data about the atom from the PDB file, the readSTRIDE method pulls secondary features from data compiled by the STRIDE program based on the residue values of the atoms. 
  
  Once the list of atoms has been created, it is split up by element type in the split_atoms_by_element_type method. This is done so that the rigidity indices (which are based on element pair interactions) can be computed. For each atom, this results in the creation of 9 features to be used in the RF/GBT models, as well as 3 (8,30) "image" feature inputs for the CNN model. 
  
</details>

<details>
  <summary>Extracting the Data</summary>
  
  ## PDB Files and Features Used
  The training and test data for this project comes from the protein databank in the form of PDB files. These are plaintext files which contain information obtained through xray crystallography about proteins. There are global features, which apply to all atoms within a protein, and local features which are dependent on the atom. Examples of global features used are the resolution (gives a notion of the quality of the protein model) and the number of heavy atoms (gives a notion of the size of the protein). The construction of local features which represent local rigidity of the structure is where a lot of the work of the paper lies. The idea of Multi Weighted Colored Graphs (MWCGs) are used to generate a rigidity index for an atom based on the position and element type pair interactions, which are included in the PDB files. Also included are 12 secondary features which are generated from a program called STRIDE, which categorize atoms as belonging to sub structures such as helixes or coils. The residue number of an atom obtained from the PDB file determines which atoms should share secondary features.
  
  An atom class is included in feature_compiler.py, which contains the following class variables:
  
  ```
  pos = position data (x,y,z) (obtained directly from PDB)
  res_number = which residue the atom belongs to
  amino_type = amino type of atom in one hot format, 20 choices (obtained directly from PDB)
  amino types are:
  ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
  'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
  heavy_type = heavy element type of atom in one hot format, 5 choices (obtained directly from PDB)
  heavy element types are:
  ['C', 'N', 'O', 'S', 'H']
  atom_name = name of atom (obtained directly from PDB)
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
  ```
  
  Some of these values are pulled directly from the PDB file, but many of them are calculated using the element type and position information from the PDB file. There is also a value for atoms called the occupancy condition, which determines the probability that a specific atom will occupy a certain position. For a single position, the occupancy condition of all atoms at that position should sum to 1. For this reason, we only consider atoms with occupancy greater than .5, or when two atoms have an occupancy condition of 0.5, we only consider one of them. This is handled in the below lines:
  
  ```
  if float(line[54:60]) == 0.5:
      occupancy_condition = not occupancy_condition
      print(occupancy_condition)
  if float(line[54:60]) > 0.5 or occupancy_condition == True: 
      current_atom = atom()
  ```
  
  The readPDB method performs this check for each atom, extracts all needed features, and adds the atoms to a list. This extraction required a great amount of familiarty with the data set to deal with situations such as the occupancy condition scenario described above. To construct the global feature from the nubmer of heavy atoms, one hot encoding was used with defined cutoffs representing various size categories. After extracting the neccessary data about the atom from the PDB file, the readSTRIDE method pulls secondary features from data compiled by the STRIDE program based on the residue values of the atoms. 
  
  Once the list of atoms has been created, it is split up by element type in the split_atoms_by_element_type method. This is done so that the rigidity indices (which are based on element pair interactions) can be computed. For each atom, this results in the creation of 9 features to be used in the RF/GBT models, as well as 3 (8,30) "image" feature inputs for the CNN model. 
  
</details>
