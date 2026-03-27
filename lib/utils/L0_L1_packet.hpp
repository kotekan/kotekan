//
// The following pseudo header file isn't actually valid C++ code, but is
// intended to document the L0_L1 UDP packet format in a C++-like syntax.
//
// Here we are specifying the data packet sent from the L0 beam forming process to
// the L1 nodes to search for FRBs.  The characteristics of the system are as follows:
//
//    - Each correlator node (where the L0 process operates) processes all 1024
//      fan-beams for 4 x 390kHz-wide "coarse" frequency bands.  
// 
//    - The L0 process upchannelizes each 390kHz "coarse" frequency band to form 
//      128 x 3.05kHz subbands. In the same process it will dedisperse the data to 
//      a fiducial DM value, then downsample the data to  16 x 24kHz "fine" subbands. 
//
//    - The 4 coarse frequency bands on each correlator node are not currently considered 
//      to be contiguous/consecutive.
//
//    - The amplitude time series for each beam is then squared, polarization summed and 
//      integrated to a 1-ms timescale and stored as uint8. 
//
//    - Each of the 256 correlator nodes will send data packets to each of the frbsearch 
//      nodes (which operate the L1 processes for n beams, currently n=8).
//
//    - The number of time samples per packet is chosen to match the max ethernet packet
//      size (approx 9KB assuming Gbps ethernet with jumbo frames).  For full CHIME,
//      16 time samples per packet is a good choice.
//
//    - Thus each packet contains a 4-dimensional 8-bit array of shape (nbeams, nfreq_coarse, 
//      nupfreq, ntsamp), where:
//
//         nbeams = number of beams processed by each L1 node (currently 8)
//         nfreq_coarse = number of coarse frequencies processed by each L0 node (currently 4)
//         nupfreq = upchannelization channel (currently 16)
//         ntsamp = number of time samples (currently 16)
//
//     - The packets are sent via UDP to the frbsearch node and some fraction may drop along the way.
//
//     - Optionally, the packet may allow for data compression using bitshuffle
//        (https://github.com/kiyo-masui/bitshuffle).
//
// We don't bother putting integer data in network byte order, or converting floating-point data to
// an external representation, since we're assuming that all nodes are Intel CPUs with little-endian
// ints and IEEE 754 floats.
//
// The packet header size in bytes is currently
//    32 + 2*nbeam + 2*nfreq_coarse + 8*nbeam*nfreq
//
// For full CHIME, we expect to use (nbeam, nfreq_coarse, nupfreq, ntsamp) = (8, 4, 16, 16)
// which gives a 304-byte header and an 8192-byte data segment.
//
// For the pathfinder, I'm currently using  (nbeam, nfreq_coarse, nupfreq, ntsamp) = (1, 32, 1, 256) 
// which gives a 346-byte header and an 8192-byte data segment.
//

struct L0_L1_header {
    // This file describes protocol version 2.
    // A 32-bit protocol number is overkill, but means that all fields below are aligned on their
    // "natural" boundaries (i.e. fields with size Nbytes have byte offsets which are multiples of Nbytes)
    uint32_t    protocol_version;

    // This is a signed 16 bit integer which  gives the total size of the 'data' array below in bytes. 
    // Negative sizes indicate that bitshuffle compression has been applied and in this case, the 
    // absolute value indicates the size of the array.  If data_nbytes is positive, it should always
    // be equal to (nbeam * nfreq_coarse * nupfreq * ntsamp).
    int16_t     data_nbytes;

    // This is the duration of a time sample, in FPGA counts.
    // The duration in seconds is dt = (2.56e-6 * fpga_counts_per_sample)
    uint16_t    fpga_counts_per_sample;

    // This is the time in nanoseconds since Unix epoch of the first FPGA sample (fpga_count=0)
    uint64_t    fpga_frame0_ns;

    // This is the time index (in FPGA counts) of the first time sample in the packet.
    // The packet sender is responsible for "unwrapping" the 32-bit FPGA timestamp to 64 bits.
    // Must be divisible by fpga_counts_per_sample.
    uint64_t    fpga_count;

    uint16_t    nbeam;          // number of beams in packet
    uint16_t    nfreq_coarse;   // number of "FPGA" frequency channels between 0 and 1024
    uint16_t    nupfreq;        // upsampling factor (probably either 1 or 16)
    uint16_t    ntsamp;         // number of time samples in packet

    // Beam IDs in the data packet, numbered from 0-1023. 
    // It is not assumed that the 8 beams processed on each frbsearch node are consecutive. 
    // The mapping between beam ids and sky locations is currently undefined.
    uint16_t    beam_ids[nbeam];

    // Coarse frequency ids (i.e. 390 kHZ), between 0-1023, not assumed consecutive.
    // The frequencies are assumed ordered from highest to lowest, i.e. index zero is 800 MHz.
    uint16_t    coarse_freq_ids[nfreq_coarse];
    
    // The 'scale' and 'offset' parameters are used for 8-bit encoding of intensities:
    //    Floating-point intensity = scale * (raw 8-bit value) + offset
    //
    // A design decision: what granularity should the scale and offset parameters have?
    // In this version of the protocol, we use assume that they can depend on beam and
    // coarse frequency index, but are the same for each of the 16 upchannelized frequencies
    // comprising a coarse frequency, and are the same for each of the 16 time samples in
    // a packet.  Thus the scale of offset arrays have shape (nbeam, nfreq_coarse).

    float32     scale[nbeam * nfreq_coarse];   // byte offset (32 + 2*nbeam + 2*nfreq_coarse)
    float32     offset[nbeam * nfreq_coarse];  // byte offset (32 + 2*nbeam + 2*nfreq_coarse + 2*nbeam*nfreq_coarse)

    // The uncompressed data array has shape (nbeam, nfreq_coarse, nupfreq, ntsamp),
    // ordered so that the fastest changing index is the time dimension, i.e. each
    // timeseries is contiguous.  Compression will change the size of the array, but
    // the ordering will be the same after decompression.
    //
    // Thus in full chime, the uncompressed data array has shape
    //
    //  uint8 data[8][4][16][16]
    //             |  |   |   |
    //             |  |   |   +-- 16x 1-ms time samples
    //             |  |   +------ 16x 24 kHz frequency channels 
    //             |  +---------- 4x frequency bands (may not be consecutive in frequency)
    //             +------------- 8x on-sky beams
    //
    // We need a sentinel value to indicate "this entry in the array is invalid".
    // We currently take both endpoint values to be sentinels, i.e. if a byte is either
    // 0x00 or 0xff, it should be treated as masked.

    uint8_t     data[ abs(data_nbytes) ];
};
