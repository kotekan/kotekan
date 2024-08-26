import numpy as np
import zarr



def uint8tocomplexi4(u8):
    lo = u8 & 0x0f
    hi = (u8 >> 4) & 0x0f
    if lo >= 8:
        lo = 16 - lo
    if hi >= 8:
        hi = 16 - hi
    return lo + 1j * hi
vectorized_uint8tocomplexi4 = np.vectorize(uint8tocomplexi4, [np.complex64])

def uint16tofloat16(u8):
    lo = u8 & 0x0f
    hi = (u8 >> 4) & 0x0f
    if lo >= 8:
        lo = 16 - lo
    if hi >= 8:
        hi = 16 - hi
    return lo + 1j * hi
vectorized_uint8tocomplexi4 = np.vectorize(uint8tocomplexi4, [np.complex64])



g = zarr.open_group('/localhome/eschnett/data/fengine_test_pathfinder/indigo_dish_positions.00000000.zarr', mode='r')
print(g.info)

attributes = [a for a in g.attrs]
chord_metadata_version = g.attrs['chord_metadata_version']
ndishes = g.attrs['ndishes']
type_ = g.attrs['type']
dish_index = np.array(g.dish_index)

dish_positions = np.array(g.dish_positions)



g = zarr.open_group('/localhome/eschnett/data/fengine_test_pathfinder/indigo_voltage.00000000.zarr', mode='r')
print(g.info)

attributes = [a for a in g.attrs]

chord_metadata_version = g.attrs['chord_metadata_version']
coarse_freq = g.attrs['coarse_freq']
freq_upchan_factor = g.attrs['freq_upchan_factor']
half_fpga_sample0 = g.attrs['half_fpga_sample0']
ndishes = g.attrs['ndishes']
nfreq = g.attrs['nfreq']
sample0_offset = g.attrs['sample0_offset']
time_downsampling_fpga = g.attrs['time_downsampling_fpga']
type_ = g.attrs['type']
dish_index = np.array(g.dish_index)

# voltage = np.array(g.E)
voltage = vectorized_uint8tocomplexi4(np.array(g.E))



g1 = zarr.open_group('/localhome/eschnett/data/fengine_test_pathfinder.3/indigo_frb3_beams_meanstd.00000000.zarr', mode='r')
g2 = zarr.open_group('/localhome/eschnett/data/fengine_test_pathfinder/indigo_frb3_beams_meanstd.00000000.zarr', mode='r')

meanstd1 = np.array(g1.I3meanstd)
meanstd2 = np.array(g2.I3meanstd)
# meanstd1 = np.array(g1.I3meanstd).view(np.float16)
# meanstd2 = np.array(g2.I3meanstd).view(np.float16)
