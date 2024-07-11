# Configuration name
const setup = :chime

# Time between time samples
const sampling_time_μsec = 4096 / 1600

# Number of complex number components
const C = 2

# Number of polarizations
const P = 2

# Number of dishes
const D = 1024

# Number of coarse frequencies per GPU
const F = 16

# Number of time samples per frame processed by the GPU, times the Kotekan buffer depth
const T = 4 * 16384

# Maximum number of coarse frequencies per upchannelization factor, rounded up to a power of 2
const F_per_U = Dict(1 => 1, 2 => 1, 4 => 1, 8 => 1, 16 => 16, 32 => 1, 64 => 1, 128 => 1)

# Maximum number of fine frequencies per GPU, for all upchannelization factors combined
const Fbar_out = 256

# FRB downsampling factor for U=1
const Tds_U1 = 400
