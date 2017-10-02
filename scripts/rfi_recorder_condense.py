import numpy as np
import os

rfi_file_list = []
for f in os.listdir("."):
    if f.endswith(".rfi"):
        rfi_file_list.append(f)
data = np.empty([0])
for f in rfi_file_list:
	data = np.append(data, np.loadtxt(f,delimiter =','))
np.save("rfi_recorder_data_condensed",data)
