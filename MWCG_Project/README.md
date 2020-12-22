<details>
  <summary>Project Overview</summary>
  
  ## Intro
  Hello! The purpose of this project was to reproduce the predictive models and B-factor prediction accuracies achieved in *Blind Prediction of Protein B-factor and Flexibility* by David Bramer and Guo-Wei Wei. A PDF of the paper is included in the repo. The code used to reproduce the results is meant to run on MSU's high performance computing cluster, so instead of showing you how to run it yourself I will walk you through an overview of the code and methods used. A less detailed summary is also included below.
  
  ## Summary
  This project involved becoming familiar with a complicated dataset from the protein databank so that feature extraction could be performed properly. This data comes in the form of PDB files, which contain information about each atom in a protein. 364 proteins were considered, with over 600,000 B-factors predicted in total. Secondary features were also found using the STRIDE software. Finally, new features were generated using the multi weighted colored graph method described in the paper, which involves considering element pair interactions and applying "Lorentz and exponential radial basis functions at various scales to construct multi scale features" (Bramer, section 3 of *Blind Prediction of Protein B-factor and Flexibility*), including "images" to be used as input for a convolutional neural network (CNN).
  
  After feature extraction and generation is complete, a random forest, gradient boosted decision tree, and CNN model are used for regression to predict the B-factors of each atom. The leave one out method is used, where all atoms in a single protein are predicted using all atoms from the other 363 proteins. Subsets of the full dataset containing small, medium, and large proteins were also predicted, as well as predictions on only the alpha carbon atoms. The pearson correlation coefficient was used to measure accuracy, and was reported both for each protein but also averaged across the full dataset and subsets.
  
  Due to the size of the dataset and the fact that the leave one out method was used, this project involved employing tricks such as using subsets of the data to tune the model and splitting the data into ~10 groups for cross validation instead of performing the full leave one out prediction process. Once the models were tuned and working properly, the methods in the paper were employed. I also used pickle to save outputs of the feature extraction and prediction modules, so that I could separate the workflow into more manageable pieces. 
  
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
  
  The readPDB method performs this check for each atom, extracts all needed features, and adds the atoms to a list. This extraction required a great amount of familiarty with the data set to deal with situations such as the occupancy condition scenario described above. To construct the global feature from the number of heavy atoms, one hot encoding was used with defined cutoffs representing various size categories. After extracting the neccessary data about the atom from the PDB file, the readSTRIDE method pulls secondary features from data compiled by the STRIDE program based on the residue values of the atoms. 
  
  Once the list of atoms has been created, it is split up by element type in the ```split_atoms_by_element_type``` method. This is done so that the rigidity indices (which are based on element pair interactions) can be computed. For each atom, this results in the creation of 9 features to be used in the RF/GBT models, as well as 3 (8,30) "image" feature inputs for the CNN model. 
  
</details>

<details>
  <summary>Methods and Models Used</summary>
  
  ## Prediction Methods and Metrics
  The one vs all method was used to predict the B-factors of atoms in each protein. For each protein, the atoms of all of the other 363 proteins were used as training data, and the atoms of the target protein were predicted. There were also subsets of the 364 proteins containing small, medium, and large proteins, on which predictions were made via the same method. Predictions on only the alpha carbon atoms, using the same training sets, were also made. The pearson correlation coefficient was used to measure the accuracy in this regression task. One drawback of this method is the extreme amount of time it takes (even with parallelized code, it could take hours for one prediction, and more realistically days based on the queue time for the submitted jobs). For this reason, while I was testing and tuning the models, I often split the proteins into ~10 groups instead. I used the pickle library to condense and store the features generated in feature_compiler.py, so that features did not have to be re-generated for each test of the models. These were stored in dictionaries to keep track of the protein for each sample and seperate the input for the CNN from the other features. 
  
  ```
  combined_feature_dict = {'protein_ID': self.protein_ID, 'features': combined_feature_array,\
            'CNN_input': CNN_input}
  ```
  
  ## Preparing the Data 
  predictor.py starts by loading the pickled feature dictionaries, and then splitting them into the training and test data suitable for the GBT/RF models:
  
  ```
  all_dicts = copy.deepcopy(all_proteins_dict_list)
  popped_dict = all_dicts.pop(index) # pulls out the protein we are predicting in this this run
  popped_features = popped_dict['features'] # retrieves the features for this protein
  remaining_features = []
  for dictionary in all_dicts:
      remaining_features.append(dictionary['features']) # retrieves the features for the proteins to be used for training
  . . .
  # splits the labels off of the feature lists
  X_test = popped_features[:, 0:59] 
  y_test = popped_features[:, 59]
  popped_features_CA = popped_features[popped_features[:, 60] == 1]
  X_test_CA = popped_features_CA[:, 0:59]
  y_test_CA = popped_features_CA[:, 59]
  remaining_features = np.concatenate(remaining_features, axis=0)
  print(remaining_features.shape)
  X_train = remaining_features[:, 0:59]
  y_train = remaining_features[:, 59]
  ```
  
  ## Gradient Boosted Trees and Random Forest
  predictor.py then uses gradient boosted tree and random forest methods to predict the b factor for the atoms in the chosen protein. The sklearn library was used to employ both of these models:
  
  ```
  GBT_reg = GBT(loss = 'ls', n_estimators = 1600, learning_rate = 0.008, \
           max_depth = 4, min_samples_leaf = 9, min_samples_split = 9).fit(X_train, y_train)
  GBT_ypred = GBT_reg.predict(X_test)
  . . .
  RF_reg = RF(n_estimators = 500).fit(X_train, y_train)
  RF_ypred = RF_reg.predict(X_test)
  ```
  
  Predictions are also made for the subset of alpha carbon atoms, and the Pearson correlation coefficient is calculated for each prediction and stored in a dictionary along with the protein ID:
  
  ```
  GBT_dict = {'protein_ID': current_ID, 'pCC': GBT_pCC, 'pCC_CA': GBT_pCC_CA}
  RF_dict = {'protein_ID': current_ID, 'pCC': RF_pCC, 'pCC_CA': RF_pCC_CA}
  ```
  
  These results are then pickled so that they can be averaged across relevant protein groups and displayed in pcc.py.
  
  ## Convolutional Neural Network
  The rigidity index "image" is first normalized and used as an input for a CNN. Then the output is concatenated with the other features and used as input for a traditional neural network. This model was built using Keras with the tensorflow backend and consists of two convolution layers followed by a dropout layer, and a dense layer, with the activation function for all layers being a leaky RELU:
  
  ```
  CNN_in = Input(shape = (8, 30, 3))
  CNN_1 = Conv2D(filters = 14, kernel_size=2)(CNN_in) 
  CNN_act_1 = LeakyReLU(alpha = 0.01)(CNN_1)
  CNN_2 = Conv2D(filters = 16, kernel_size=2)(CNN_act_1) 
  CNN_act_2 = LeakyReLU(alpha = 0.01)(CNN_2)
  CNN_drop_1 = Dropout(0.5)(CNN_act_2)
  CNN_dense_1 = Dense(59)(CNN_drop_1) 
  CNN_act_3 = LeakyReLU(alpha = 0.01)(CNN_dense_1)
  CNN_out = Flatten()(CNN_act_3)
  CNN_model = Model(CNN_in, CNN_out)
  ```
  
  After concatenating the flattened output with the other features, the data is pushed through this neural network:
  
  ```
  merged_dense_1 = Dense(100)(concatenated)
  merged_act_1 = LeakyReLU(alpha = 0.01)(merged_dense_1)
  merged_drop_1 = Dropout(0.5)(merged_act_1)
  merged_dense_2 = Dense(10)(merged_drop_1)
  merged_act_2 = LeakyReLU(alpha = 0.01)(merged_dense_2)
  merged_drop_2 = Dropout(0.25)(merged_act_2)
  merged_out = Dense(1)(merged_drop_2)
  final_model = Model([CNN_in, fea_in], merged_out)
  ```
  
</details>
