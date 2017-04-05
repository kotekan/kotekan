import numpy as np
import matplotlib.pyplot as plt
import sys
import time

header_length = np.fromfile(sys.stdin, dtype=np.uint32, count=1, sep='')
header = np.fromfile(sys.stdin, dtype=np.uint32, count=header_length, sep='')

data_length = header[0]
data_dims = header[1:4]




data = np.fromfile(sys.stdin, dtype=np.int32, count=data_length, sep='')
data=data.reshape(data_dims)

corr_matrix = np.zeros([2048,2048,2])
x_blk=0
y_blk=0
for i in np.arange(data_dims[0]):
	corr_matrix[x_blk:x_blk+32,y_blk:y_blk+32,:]=data[i,:,:].reshape([32,32,2])
	x_blk+=32
	if x_blk >= 2048:
		y_blk+=32;
		x_blk=y_blk;

#plt.figure(figsize=[12,9])
#plt.imshow(corr_matrix.T,interpolation='none')
#plt.colorbar()
#plt.title("Correlation Matrix (Re)")
#plt.xlabel("X Input")
#plt.ylabel("Y Input")

f, (ax1, ax2) = plt.subplots(1, 2, figsize=[18,6])

im=ax1.imshow(corr_matrix[:,:,0].T,origin='upper',interpolation='nearest')
ax1.set_title("Re(Correlation Matrix)")
f.colorbar(im,ax=ax1)
im=ax2.imshow(corr_matrix[:,:,1].T,origin='upper',interpolation='nearest')
ax2.set_title("Im(Correlation Matrix)")

f.colorbar(im,ax=ax2)

ax1.set_xlabel("X Input")
ax2.set_xlabel("X Input")
ax1.set_ylabel("Y Input")

vsize=ax1.get_position().size[1]  #fraction of figure occupied by axes
axesdpi= int(2048/(f.get_size_inches()[1]*ax1.get_position().size[1])) 

plt.savefig("correlation_{}.pdf".format(time.time()), dpi=axesdpi,bbox_inches='tight')








'''
target = open("result", 'w')
target.write("Start!\n")
target.write("{}\n".format(data_length))
target.write("{}\n".format(data_dims))
target.write("{}\n".format(data.shape));
target.write("{}\n".format(np.amin(data)));
target.write("{}\n".format(np.mean(np.mean(data,axis=0),axis=0)[0]))
target.write("{}\n".format(np.mean(np.mean(data,axis=0),axis=0)[1]))
target.write("Stop!\n")
target.close()
'''


# call with e.g. curl http://localhost:12048/plot_corr_matrix -X POST -d '{}'