import numpy as np
from time import time
#import matplotlib
#matplotlib.use('TkAgg')
import pylab
import pyopencl as cl
pylab.ion()


######
#Initialize GPU
######

#Get the platform, not sure how this could be more than one thing
plat = cl.get_platforms()[0]

#Get the devices on the platform
devs = plat.get_devices()

#Get the second device, need to change this to look for specific card etc.
dev1 = devs[0]

#create a context for the device
ctx = cl.Context(devices=[dev1])

#Create a queue for the device
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

#load data file
data_block = np.fromfile('block2.dat', dtype=np.int8)
data_block_size = data_block.size

######
#Create Buffers
######
mf = cl.mem_flags

#Test time to push to card
mult=4
rand_data = np.random.randint(256,size=data_block_size*mult).astype(np.int8)
t1 = time()
input_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rand_data )
push_time = time()-t1

print "time to initiate to card: {0}".format(push_time)

t1 = time()
cl.enqueue_copy(queue, input_buffer, rand_data)
push_time = time()-t1
print "first time to send to card: {0}".format(push_time)

transfer_times = []

for i in np.arange(3):
	rand_data = np.random.randint(256,size=data_block_size*mult).astype(np.int8)
	#input_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=rand_data )
	print rand_data.size
	#input_buffer.release()
	t1 = time()
	cl.enqueue_copy(queue, input_buffer, rand_data)
	push_time = time()-t1
	transfer_times.append(push_time)
	print "time to send to card: {0}".format(push_time)

mean_transfer = np.mean(transfer_times)

gbps = 8*data_block_size*mult/mean_transfer/1e9

print "Tranfer time from memory to GPU is {0:.5}s, corresponding to {1:.5} Gbit/s".format(mean_transfer, gbps)

######
#test transpose
######
t1 = time()
mem_data = np.zeros((data_block_size,8), dtype=np.int8)
mem_time = time()-t1

gbps = 8*mem_data.size/mem_time/1e9

print "Time to create zeroed block of memory {0:.5}s, {1:.5} Gbit/s".format(mem_time, gbps)

t1 = time()
mem_cpy = mem_data.copy()
mem_time = time() - t1
gbps = 8*mem_cpy.size/mem_time/1e9
print "Time to copy block of memory {0:.5}s, {1:.5} Gbit/s".format(mem_time, gbps)

t1 = time()
mem_transpose = mem_data.transpose().copy()
mem_time = time() - t1
gbps = 8*mem_data.size/mem_time/1e9
print "Time to transpose block of memory {0:.5}s, {1:.5} Gbit/s".format(mem_time, gbps)