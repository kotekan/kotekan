# Number of PFB taps `M` of the upchannelizers (`upchan.jl`). The FRB beamformer (`frb.jl`) needs
# it for the time offset of the upchannelized voltages. Must match
# `kotekan::upchan_default_num_taps` in `lib/cuda/upchannelizeReference.hpp`.
const upchan_number_of_taps = 4
