import sys
import numpy as np
import matplotlib.pyplot as plt

print("Reading file: {}".format(sys.argv[1]))

metadata_t = np.dtype([('metadata_size', np.uint32), ('fpga_seq_num', np.int64),  ('gps_time', [('', np.int64),('', np.int64)]), ('gps_time_flag', np.uint32), ('freq_bin_num', np.uint32), ('norm_frac', np.float32), ('num_samples_integrated', np.uint32), ('num_samples_expected', np.uint32), ('compressed_data_size', np.uint32) ])

dt = np.dtype([ ('metadata', metadata_t), ('data', np.float32, (1024 * 128,)) ])

data = np.fromfile(sys.argv[1], dtype=dt)

print "Metadata"
print "--------"
print data['metadata']

print data['data']
plt.plot(data['data'][0])
plt.show()
