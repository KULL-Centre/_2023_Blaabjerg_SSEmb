import json # Opening JSON filef = open('chain_set_splits.json',)data = json.load(f)# Split and process#pdbids_train = [x[:-2].upper() for x in data["train"]]pdbids_train = [x[:-2].upper() for x in data["train"]]pdbids_val = [x[:-2].upper() for x in data["validation"]]pdbids_test = [x[:-2].upper() for x in data["test"]]# Write# tf = open("CATH_pdbids_train.txt", "w")# for element in pdbids_train:#     tf. write(element + "\n")# tf.close()tf = open("CATH_pdbids_train.txt", "w")for element in pdbids_train:    tf. write(element + "\n")tf.close()tf = open("CATH_pdbids_val.txt", "w")for element in pdbids_val:    tf. write(element + "\n")tf.close()tf = open("CATH_pdbids_test.txt", "w")for element in pdbids_test:    tf. write(element + "\n")tf.close()