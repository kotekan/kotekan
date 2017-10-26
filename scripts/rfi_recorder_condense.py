import numpy as np
import os
import matplotlib.pyplot as plt

"""
rfi_file_list = []
for f in os.listdir("."):
    if f.endswith(".rfi"):
        rfi_file_list.append(f)
data = np.empty([0])
for f in rfi_file_list:
	data = np.append(data, np.loadtxt(f,delimiter =','))
np.save("rfi_recorder_data_condensed",data)
"""

def find_first_last_seq_num(data_list):
	min_seq = np.min(data_list[0]['seq'])
	max_seq = 0
	for data in data_list:
		max_seq = np.max(data['seq'])*(np.max(data['seq']) > max_seq)
		min_seq = np.min(data['seq'])*(np.min(data['seq']) < min_seq)	
	return (min_seq,max_seq)

#Locate all the files
rfi_file_list = []
for f in os.listdir("."): 
    if f.endswith(".rfi"):
        rfi_file_list.append(f)

#Extract All the data
data_list = []
for f in rfi_file_list:
	data = np.fromfile(f,dtype=np.dtype([('bin', 'i4',1), ('seq', 'i8',1), ('mask', 'f4',1)]))
	data_list.append(data)
	#data['seq'] = np.arange(data['seq'].size)

#Save into data array
min_seq, max_seq = find_first_last_seq_num(data_list)
Total_data = np.zeros([1024,max_seq-min_seq+1])
print(Total_data.shape)
for data in data_list:
	Total_data[data['bin'],data['seq']-min_seq] = data['mask']
#save to file
np.save("RFI_Time_Mask_Waterfall",Total_data)
