# Configuration name
const setup = :pathfinder

# Time between time samples
const sampling_time_μsec = 16384 / 3200

# Number of complex number components
const C = 2

# Number of polarizations
const P = 2

# Number of dishes
const D = 64

# Number of coarse frequencies per GPU
const F = 384

# Number of time samples per frame processed by the GPU, times the Kotekan buffer depth
const T = 4 * 8192

# Maximum number of coarse frequencies per upchannelization factor, rounded up to a power of 2
const F_per_U = Dict(1 => 128, 2 => 128, 4 => 128, 8 => 64, 16 => 64, 32 => 64, 64 => 32)

# Maximum number of fine frequencies per GPU, for all upchannelization factors combined
const Fbar_out = 4096

# FRB downsampling factor for U=1
const Tds_U1 = 192
