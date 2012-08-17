import pyopencl as cl
import numpy as np
from time import time

N_ANT = 256
N_ITER = 128*1024

#read in test data
data_block = np.fromfile('block.dat', dtype=np.int8)


#Get the platform, not sure how this could be more than one thing
plat = cl.get_platforms()[0]

#Get the devices on the platform
devs = plat.get_devices()

#Get the second device, need to change this to look for specific card etc.
dev1 = devs[1]

#create a context for the device
ctx = cl.Context(devices=[dev1])

#Create a queue for the device
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

#Load the kernel into a string from file
KERNEL_CODE = open('test.cl', 'r').read()

#Get infor about the device, and calculate potential performance
local_mem = dev1.local_mem_size
max_clock_frequency = dev1.max_clock_frequency
max_compute_units = dev1.max_compute_units
#need to figure out why each compute unit does 
theoretical_perf = max_clock_frequency*1e6 * max_compute_units * 16 * 4 *2/1e12
print "Theoretical Maximum Performance ", theoretical_perf, " TFLOPs"

#not sure how to use this yet
#kernel_params = {"packed": packed,
#		"corr_buff":corr_buff, "id_x_map":id_x_map, "id_y_map":id_y_map}

#Create program and kernel
corr_program = cl.Program(ctx, KERNEL_CODE).build()
corr_kernel = corr_program.corr

######
#Create Buffers
######
mf = cl.mem_flags
input_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_block )

s1_blk = 32
n_blk = (N_ANT / s1_blk) * (N_ANT / s1_blk + 1) / 2.
zeros = np.zeros(n_blk*(s1_blk*s1_blk)*2, dtype=np.int32)

corr_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros)
#corr_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, size=n_blk)???

#too lazy to make this better and actually figure out what I'm writing.  Is calculating
# the blocks to share kernels efficiently.
global_id_x_map = np.zeros(n_blk, dtype=np.uint)
global_id_y_map = np.zeros(n_blk, dtype=np.uint)
for i in np.arange(n_blk):
    t = (np.sqrt(1 + 8*(n_blk-i-1))-1)/2
    y = N_ANT/s1_blk-t-1
    x = (t+1)*(t+2)/2 + (i - n_blk)+y
    global_id_x_map[i] = x
    global_id_y_map[i] = y

id_x_map = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=global_id_x_map)
id_y_map = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=global_id_y_map)

unpacked = cl.LocalMemory(4*16*4*4)

event = corr_kernel(queue, zeros.shape[::-1], input_buffer, corr_buffer, id_x_map, id_y_map, unpacked )
event.wait()

cl.enqueue_copy(queue, zeros, corr_buffer)

print zeros
		