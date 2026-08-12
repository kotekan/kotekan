# Configuration name
const setup = :charts
# `compute_capability` and `ptx_compat` select the PTX we *emit* (`.target` and
# `.version`). LLVM 18.1.7 (Julia 1.12) supports `cap <= 9.0` and `ptx <= 8.3`;
# `.target sm_120` would need `ptx >= 8.7`, i.e. LLVM 20 / Julia 1.13. Note that
# CUDA.jl silently clamps an unsupported `cap` instead of complaining.
# This is not a limitation in practice: `.target sm_86` PTX can be compiled for
# any `sm_MN` with `MN >= 86`, and ptxas then generates native SASS for it.
const compute_capability = v"8.6" # A40, the GPU generating the kernels
const ptx_compat = v"8.0"
# `cuda_arch` is the GPU we *run* on. It is passed to ptxas (`--gpu-name`) when
# Kotekan compiles the PTX at startup.
const cuda_arch = "sm_120" # RTX 5090

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
const T = 4 * 16384

# Maximum number of coarse frequencies per upchannelization factor, rounded up to a power of 2
const F_per_U = Dict(1 => 1, 2 => 1, 4 => 1, 8 => 1, 16 => 1, 32 => 336, 64 => 1, 128 => 1)

# Maximum number of fine frequencies per GPU, for all upchannelization factors combined
const Fbar_out = 10752

# FRB downsampling factor for U=1
const Tds_U1 = 320
