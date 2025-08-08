using HDF5

include("../lib/stages/FEngine.jl")

source_amplitude = 1.0
source_frequency = 0.3e+9
source_position_sinx = 0.02
source_position_siny = 0.03
dish_separation_x = 6.3
dish_separation_y = 8.5
ndishes_i = 32
ndishes_j = 16
adc_frequency = 3.0e+9
ntaps = 4
nfreq = 16
ntimes = 32768

FEngine.setup(
    source_amplitude,
    source_frequency,
    source_position_sinx,
    source_position_siny,
    dish_separation_x,
    dish_separation_y,
    ndishes_i,
    ndishes_j,
    adc_frequency,
    ntaps,
    nfreq,
    ntimes,
)

sz = 536870912
ndishes = 512
nfreqs = nfreq
npolrs = 2
# ntimes = 32768
frame_index = 1
E = zeros(UInt8, sz);

FEngine.set_E(pointer(E), sz, ndishes, nfreqs, npolrs, ntimes, frame_index)
