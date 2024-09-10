# Configuration name
const setup = :chord

# Time between time samples
const sampling_time_μsec = 16384 / 3200

# Number of complex number components
const C = 2

# Number of polarizations
const P = 2

# Number of dishes
const D = 512

# Number of coarse frequencies per GPU
const F = 48

# Number of time samples per frame processed by the GPU, times the Kotekan buffer depth
const T = 4 * 8192

# Maximum number of coarse frequencies per upchannelization factor, rounded up to a power of 2
const F_per_U = Dict(1 => 16, 2 => 16, 4 => 16, 8 => 8, 16 => 8, 32 => 8, 64 => 4, 128 => 1)

# Maximum number of fine frequencies per GPU, for all upchannelization factors combined
const Fbar_out = 512

# FRB downsampling factor for U=1
const Tds_U1 = 192
