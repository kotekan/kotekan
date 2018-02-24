import numpy as np
import sys
import pickle
import struct

INF = sys.argv[1] #.npy files from Deborah
print "Reading in File", INF, '--------------'

data = np.load(INF)  
print "data[0]=", data[0], "data[204]=",data[204], "data[205]=", data[205]

#If reordering from correlator order
'''
with open('../Re-ordering/feed_corr_map.pkl','rb') as f:
    data_map = pickle.load(f)
reorder = np.zeros([2,2048],dtype='int')
for i in range(2048):
    reorder[0][i] = i
    reorder[1][i] = data_map[i]["corr_based_id"]  #This is the jambled order
New = np.zeros(2048,dtype='int')
for i in range(2048):
    New[i] =  np.where(reorder[1,:] == i)[0]
'''

#If reordering from cylinder order
New2 = np.zeros(2048,dtype='int')
for i in range(1024):
    newID = (i & 256 ) + (i & 512) + i
    New2[i] = newID
    New2[i+1024] = newID+256


data2 = np.zeros(2048,dtype=complex)
#data2 = data[New]
data2 = data[New2]

#Write out as binary
freq = float(INF.split(".npy")[0].split("_")[-1])
bin = int((800.0 - freq)*1024.0/400.0)
print "freq=", freq, "bin=",bin

outbase = "quick_gains_"+str(bin).zfill(4)+"_reordered"
flattend = [f for sublist in ((c.real, c.imag) for c in data2) for f in sublist]
output_file = open(outbase+'.bin', 'wb')
flattendarray = np.array(flattend)
s = struct.pack('f'*len(flattendarray), *flattendarray)
output_file.write(s)
output_file.close()

