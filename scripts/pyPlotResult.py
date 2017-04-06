import numpy as np
import matplotlib.pyplot as plt
import sys
import time

header_length = np.fromfile(sys.stdin, dtype=np.uint32, count=1, sep='')
header = np.fromfile(sys.stdin, dtype=np.uint32, count=header_length, sep='')

data_length = header[0]
data_type = header[1]

if (data_type == 0x12340001): #N^2
	num_elements = header[2]
	block_dims = header[3:6]
	num_blocks = (num_elements/block_dims[0]) * (num_elements/block_dims[0]+1)/2

	target = open("result", 'w')
	target.write("Start!\n")
	target.write("Data Length: {}\n".format(data_length))
	target.write("Elements: {}\n".format(num_elements))
	target.write("Block Dims: {}\n".format(block_dims))
	target.write("Blocks: {}\n".format(num_blocks))
	target.close()


	data = np.fromfile(sys.stdin, dtype=np.int32, count=data_length, sep='')
	data=data.reshape([num_blocks,block_dims[0]*block_dims[1],2])

	corr_matrix = np.zeros([num_elements,num_elements,2])
	x_blk=0
	y_blk=0
	for i in np.arange(num_blocks):
		corr_matrix[x_blk:x_blk+block_dims[0],y_blk:y_blk+block_dims[1],:]=data[i,:,:].reshape(block_dims)
		x_blk+=block_dims[0]
		if x_blk >= num_elements:
			y_blk+=block_dims[1];
			x_blk=y_blk;

	f, (ax1, ax2) = plt.subplots(1, 2, figsize=[18,6])

	im=ax1.imshow(corr_matrix[:,:,1].T,origin='upper',interpolation='nearest')
	ax1.set_title("Re(Correlation Matrix)")
	f.colorbar(im,ax=ax1)

	im=ax2.imshow(corr_matrix[:,:,0].T,origin='upper',interpolation='nearest')
	ax2.set_title("Im(Correlation Matrix)")
	f.colorbar(im,ax=ax2)

	ax1.set_xlabel("X Input")
	ax2.set_xlabel("X Input")
	ax1.set_ylabel("Y Input")

	vsize=ax1.get_position().size[1]  #fraction of figure occupied by axes
	axesdpi= int(2048/(f.get_size_inches()[1]*ax1.get_position().size[1])) 

	plt.savefig("correlation_{}.pdf".format(time.time()), dpi=axesdpi,bbox_inches='tight')








'''
target.write("{}\n".format(data.shape));
target.write("{}\n".format(np.amin(data)));
target.write("{}\n".format(np.mean(np.mean(data,axis=0),axis=0)[0]))
target.write("{}\n".format(np.mean(np.mean(data,axis=0),axis=0)[1]))
target.write("Stop!\n")
'''

# call with e.g. curl http://localhost:12048/plot_corr_matrix -X POST -d '{}'