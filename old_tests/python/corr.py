import pyopencl as cl
import numpy as np
from time import time
import matplotlib
matplotlib.use('TkAgg')
import pylab
#pylab.ion()

N_ANT = 256
N_ITER = 128*1024

#read in test data should make this be (128*1024, 256)
data_block = np.fromfile('block2.dat', dtype=np.int8)



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

#Load the kernel into a string from file
KERNEL_CODE = open('test.cl', 'r').read()

#Get infor about the device, and calculate potential performance
local_mem = dev1.local_mem_size
max_clock_frequency = dev1.max_clock_frequency
max_compute_units = dev1.max_compute_units
#need to figure out why each compute unit does 
theoretical_perf = max_clock_frequency*1e6 * max_compute_units * 16 * 4 *2/1e12
print "Theoretical Maximum Performance ", theoretical_perf, " TFLOPs"


#Create program and kernel
corr_program = cl.Program(ctx, KERNEL_CODE).build()
corr_kernel = corr_program.corr

######
#Create Buffers
######
mf = cl.mem_flags
input_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_block )

#size of one block is 32x32
s1_blk = 32
#number of blocks is n(n+1)/2 where n is number of antennas divided by block side size
n_blk = (N_ANT / s1_blk) * (N_ANT / s1_blk + 1) / 2.
#output is then number of blocks * block size * 2(real and imaginary)
zeros = np.zeros(n_blk*(s1_blk*s1_blk)*2, dtype=np.int32)

#create correlation output buffer, zero memory
corr_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=zeros)
#corr_buffer = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, size=n_blk)???

#too lazy to make this better and actually figure out what I'm writing.  Is calculating
# the blocks to share kernels efficiently. gives the x and y position of a given block.
#Commented out Keiths way, done below possibly more obvious (no square roots!)
# global_id_x_map = np.zeros(n_blk, dtype=np.uint)
# global_id_y_map = np.zeros(n_blk, dtype=np.uint)
# for i in np.arange(n_blk):
#     t = int((np.sqrt(1 + 8*(n_blk-i-1))-1)/2)
#     y = N_ANT/s1_blk-t-1
#     x = (t+1)*(t+2)/2 + (i - n_blk)+y
#     global_id_x_map[i] = x
#     global_id_y_map[i] = y
#     print t
#     print x
#     print y

global_id_x_map = np.zeros(n_blk, dtype=np.uint32)
global_id_y_map = np.zeros(n_blk, dtype=np.uint32)
cont = 0
blocks_per_side = N_ANT/s1_blk
for i in np.arange(blocks_per_side):
    for j in np.arange(i,blocks_per_side):
        global_id_x_map[cont] = j
        global_id_y_map[cont] = i
        cont += 1

id_x_map = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=global_id_x_map)
id_y_map = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=global_id_y_map)

unpacked = cl.LocalMemory(4*16*4*4)


global_work_size = [8,8,int(n_blk*N_ITER/256)]
local_work_size = [8,8,1]
corr_kernel.set_arg(0,input_buffer)
corr_kernel.set_arg(1,corr_buffer)
corr_kernel.set_arg(2,id_x_map)
corr_kernel.set_arg(3,id_y_map)
corr_kernel.set_arg(4,unpacked)

niter = 36
#warmup
warmup=1
for i in np.arange(warmup):
    event = cl.enqueue_nd_range_kernel(queue, corr_kernel, global_work_size, local_work_size, global_work_offset=None, wait_for=None, g_times_l=False)  
event.wait()
count = niter - warmup
t1 = time()
for i in np.arange(count):
    #maybe should be using cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size, global_work_offset=None, wait_for=None, g_times_l=True)
    #to be the same as keiths.
    event = cl.enqueue_nd_range_kernel(queue, corr_kernel, global_work_size, local_work_size, global_work_offset=None, wait_for=None, g_times_l=False)  
    #event = corr_kernel(queue, zeros.shape[::-1], input_buffer, corr_buffer, id_x_map, id_y_map, unpacked )

event.wait()

gpu_time = (time()-t1)
print "Unpacking rate: %6.4fs on GPU (%.1f kHz)\n" % (gpu_time,N_ITER/gpu_time/1000*count);
print "    [Theoretical max: @%.1f TFLOPS, %.1f kHz; %2.0f%% efficiency]\n" % (theoretical_perf, theoretical_perf*1e12 / (N_ANT/2.*(N_ANT+1.) * 2. * 2.) / 1e3, 100.*count*N_ITER/gpu_time / (theoretical_perf*1e12) * N_ANT/2.*(N_ANT+1.) * 2. * 2.)

  # printf("    [Theoretical max: @%.1f TFLOPS, %.1f kHz; %2.0f%% efficiency]\n",
  #  card_tflops, card_tflops*1e12 / (N_ANT/2.*(N_ANT+1.) * 2. * 2.) / 1e3,
  #  100.*nkern*N_ITER/cputime / (card_tflops*1e12) * N_ANT/2.*(N_ANT+1.) * 2. * 2.);

  # Keith's c code to do it
  # int nkern=1000;
  # unsigned int n_caccum=N_ITER/256;
  # size_t gws_corr[3]={16,8,n_blk*n_caccum};
  # size_t lws_corr[3]={16,4,1};
  # for (int i=0; i<nkern; i++)
  #   err=clEnqueueNDRangeKernel(queue, corr_kernel, 3, NULL, gws_corr, lws_corr, 0, NULL, NULL);
  # if (err) printf("Error enqueueing! %i\n",err);

## The c command
# cl_int clEnqueueNDRangeKernel ( cl_command_queue command_queue,
#     cl_kernel kernel,
#     cl_uint work_dim,
#     const size_t *global_work_offset,
#     const size_t *global_work_size,
#     const size_t *local_work_size,
#     cl_uint num_events_in_wait_list,
#     const cl_event *event_wait_list,
#     cl_event *event)


cl.enqueue_copy(queue, zeros, corr_buffer)

# print zeros.shape
# print zeros

realp = zeros[::2]
imagp = zeros[1::2]
outgpu = realp + 1.0j*imagp
outgpu.real = outgpu.real/niter
outgpu.imag = outgpu.imag/niter


#Calculate the expected output in CPU
dat = data_block.reshape((N_ITER,N_ANT))
#real 4bit data in the "low"
#imag 4bit data in the "high"
dat_real = dat & 0x0F
dat_imag = (dat >> 4) & 0x0F
dat = dat_real + 1.0j*dat_imag
t1 = time()
outcpu = np.dot(dat.conjugate().transpose(), dat).astype(np.complex64)
cpu_time = (time()-t1)
print  "Unpacking rate: %6.4fs on CPU (%.1f kHz)\n" % (cpu_time,N_ITER/cpu_time/1000)

outgpumat = np.zeros(outcpu.shape, dtype=np.complex64)


k = 0
block_total_size = s1_blk*s1_blk
errors = 0
for i in np.arange(blocks_per_side):
    for j in np.arange(i,blocks_per_side):
        dat = outgpu[k*block_total_size:(k+1)*block_total_size].reshape(s1_blk,s1_blk)
        outgpumat[i*s1_blk:(i+1)*s1_blk,j*s1_blk:(j+1)*s1_blk] = dat
        if(~(np.allclose(outcpu[i*s1_blk:(i+1)*s1_blk,j*s1_blk:(j+1)*s1_blk], dat))):
          errors +=1
        k += 1

if errors:
  print "CPU does not match GPU!!!!"		
else:
  print "CPU == GPU, success!!!"

pylab.imshow(outgpumat.real)
pylab.savefig('gpu_mat_real.png')
pylab.clf()
pylab.imshow(outcpu.real)
pylab.savefig('cpu_mat_real.png')
pylab.clf()
pylab.imshow(outgpumat.imag)
pylab.savefig('gpu_mat_imag.png')
pylab.clf()
pylab.imshow(outcpu.imag)
pylab.savefig('cpu_mat_imag.png')
pylab.clf()
