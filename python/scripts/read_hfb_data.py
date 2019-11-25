import sys
import numpy as np
import matplotlib.pyplot as plt

print("Reading file: {}".format(sys.argv[1]))

metadata_t = np.dtype([('metadata_size', np.uint32), ('fpga_seq_num', np.int64),  ('gps_time', [('s', np.int64),('ns', np.int64)]), ('gps_time_flag', np.uint32), ('freq_bin_num', np.uint32), ('norm_frac', np.float32), ('num_samples_integrated', np.uint32), ('num_samples_expected', np.uint32), ('compressed_data_size', np.uint32) ])

dt = np.dtype([ ('metadata', metadata_t), ('data', np.float32, (1024 * 128,)) ])

data = np.fromfile(sys.argv[1], dtype=dt)

print("No. of frames in the file: %d" % len(data['metadata']))

print "Metadata"
print "--------"
print metadata_t.names
print data['metadata']

#for element in data['metadata']:
#  out = ""
#  for field in metadata_t.names:
#      out = out + field + ": " + str(element[field]) + ", "
#
#  print out 

print data['data']
#plt.plot(data['data'][0])
#plt.show()
