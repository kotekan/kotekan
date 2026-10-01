# Configuration name
const setup = :selftest
# See `setup_charts.jl` for what `compute_capability`, `ptx_compat`, and
# `cuda_arch` mean.
const compute_capability = v"8.6" # A40
const ptx_compat = v"8.0"
const cuda_arch = "sm_86" # A40

# Time between time samples
const sampling_time_μsec = 16384 / 3200

# Number of complex number components
const C = 2

# Number of polarizations
const P = 2

# Number of dishes
const D = 64

# Number of coarse frequencies per GPU
const F = 4

# Number of time samples per frame processed by the GPU, times the Kotekan buffer depth
const T = 4 * 8192

# Maximum number of coarse frequencies per upchannelization factor, rounded up to a power of 2
const F_per_U = Dict(1 => 4, 2 => 4, 4 => 4, 8 => 4, 16 => 4, 32 => 4, 64 => 4, 128 => 4)

# Maximum number of fine frequencies per GPU, for all upchannelization factors combined
const Fbar_out = 512

# FRB downsampling factor for U=1
const Tds_U1 = 192
