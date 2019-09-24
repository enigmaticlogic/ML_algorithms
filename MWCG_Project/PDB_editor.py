def editPDB(PDBfile):
    with open(PDBfile) as opened_PDB:
        list_PDB = opened_PDB.readlines()
    with open('/mnt/home/storeyd3/Documents/Datasets/test/2q52.pdb', 'w') as new_file:
        for line in list_PDB:
            if 'ATOM' in line[0:7]:
                line = line.replace(line[56:60], '1.00')
            new_file.write(line)


editPDB('/mnt/home/storeyd3/Documents/Datasets/364_trim/2q52.pdb')



# also 2q4n
