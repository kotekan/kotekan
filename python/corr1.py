# example provided by Eilif Muller

from __future__ import division

KERNEL_CODE = open('testmm.cl', 'r').read()

import pyopencl as cl
from time import time
import numpy

block_size = 16

ctx = cl.create_some_context()

for dev in ctx.devices:
    assert dev.local_mem_size > 0

queue = cl.CommandQueue(ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)

#queue = cl.CommandQueue(ctx)

if False:
    a_height = 4096
    #a_height = 1024
    a_width = 2048
    #a_width = 256
    #b_height == a_width
    b_width = a_height

elif False:
    # like PyCUDA
    a_height = 2516
    a_width = 1472
    b_height = a_width
    b_width = 2144

else:
    # CL SDK
    a_width = 50*block_size
    a_height = 100*block_size
    b_width = 50*block_size
    b_height = a_width

c_width = b_width
c_height = a_height

h_a = numpy.random.rand(a_height, a_width).astype(numpy.float32)
h_b = numpy.random.rand(b_height, b_width).astype(numpy.float32)
h_c = numpy.empty((c_height, c_width)).astype(numpy.float32)


kernel_params = {"block_size": block_size,
        "w_a":a_width, "h_a":a_height, "w_b":b_width}

if "NVIDIA" in queue.device.vendor:
    options = "-cl-mad-enable -cl-fast-relaxed-math"
else:
    options = ""
prg = cl.Program(ctx, KERNEL_CODE % kernel_params,
        ).build(options=options)
kernel = prg.matrixMul
#print prg.binaries[0]

assert a_width % block_size == 0
assert a_height % block_size == 0
assert b_width % block_size == 0

# transfer host -> device -----------------------------------------------------
mf = cl.mem_flags

t1 = time()

d_a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
d_b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_b)
d_c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, size=h_c.nbytes)

push_time = time()-t1

# warmup ----------------------------------------------------------------------
for i in range(5):
    event = kernel(queue, h_c.shape[::-1], (block_size, block_size), 
            d_c_buf, d_a_buf, d_b_buf)
    event.wait()

queue.finish()

# actual benchmark ------------------------------------------------------------
t1 = time()

count = 20
for i in range(count):
    event = kernel(queue, h_c.shape[::-1], (block_size, block_size),
            d_c_buf, d_a_buf, d_b_buf)

event.wait()

gpu_time = (time()-t1)/count

# transfer device -> host -----------------------------------------------------
t1 = time()
cl.enqueue_copy(queue, h_c, d_c_buf)
pull_time = time()-t1

# timing output ---------------------------------------------------------------
gpu_total_time = gpu_time+push_time+pull_time

print "GPU push+compute+pull total [s]:", gpu_total_time
print "GPU push [s]:", push_time
print "GPU pull [s]:", pull_time
print "GPU compute (host-timed) [s]:", gpu_time
print "GPU compute (event-timed) [s]: ", (event.profile.end-event.profile.start)*1e-9

gflop = h_c.size * (a_width * 2.) / (1000**3.)
gflops = gflop / gpu_time

print
print "GFlops/s:", gflops

# cpu comparison --------------------------------------------------------------
t1 = time()
h_c_cpu = numpy.dot(h_a,h_b)
cpu_time = time()-t1

print
print "GPU==CPU:",numpy.allclose(h_c, h_c_cpu)
print
print "CPU time (s)", cpu_time
print

print "GPU speedup (with transfer): ", cpu_time/gpu_total_time
print "GPU speedup (without transfer): ", cpu_time/gpu_time

