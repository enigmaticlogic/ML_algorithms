import pandas as pd
def read_index(index_file):
    with open(index_file) as opened_index:
        list_index = opened_index.readlines()

    ID_list = []
    energy_list = []

    for line in list_index:
        split_line = line.split()
        if len(split_line) == 8:
            ID_list.append(split_line[0])
            energy_list.append(float(split_line[3]))  
            
    output_df = pd.DataFrame(list(zip(ID_list, energy_list)), columns =['ID', 'energy']) 
    return output_df