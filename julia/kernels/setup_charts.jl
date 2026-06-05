# Configuration name
const setup = :charts
const compute_capability = v"8.6" # A40
# const compute_capability = v"12.0" # RTX 5090
const ptx_compat = v"8.0"       # ???

# Time between time samples
const sampling_time_μsec = 16384 / 4915.2 # 3.333 μs

# Number of complex number components
const C = 2

# Number of polarizations
const P = 2

# Number of dishes
const D = 64

# Number of coarse frequencies per GPU
const F = 336

# Number of time samples per frame processed by the GPU, times the Kotekan buffer depth
const T = 4 * 4096

# Maximum number of coarse frequencies per upchannelization factor, rounded up to a power of 2
const F_per_U = Dict(1 => 1, 2 => 1, 4 => 1, 8 => 1, 16 => 1, 32 => 336, 64 => 1, 128 => 1)

# Maximum number of fine frequencies per GPU, for all upchannelization factors combined
const Fbar_out = 10752

# FRB downsampling factor for U=1
const Tds_U1 = 320
