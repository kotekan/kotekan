// VALIDATE THE PROBE. seed_probe.py re-implements the tracker's seed propagation in Python,
// and every "the tracker would despread at cp204 X" number this session came out of it -- so a
// silent divergence there would have sent the whole investigation after a phantom.
//
// This calls the SHIPPED gnss::propagate_seed on the tracker's own bank (GPS_L5_Q_NH, the
// 204600-chip overlaid code) with numbers given on the command line, and prints what it
// commands. Feed the same numbers to the Python and diff.
//
// usage: seedchk <cp_chips> <dop_hz> <dop_rate> <cp_rate> <ref_hop> <hop0> [prn]
#include "gnssSeedTransport.hpp"
#include "gnssChannelizedReplica.hpp"
#include "gnssSignal.hpp"
#include "pfbPrototype.hpp"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 7) {
        printf("usage: seedchk <cp_chips> <dop_hz> <dop_rate> <cp_rate> <ref_hop> <hop0> [prn]\n");
        return 2;
    }
    gnss::SeedState sd;
    sd.cp_chips = atof(argv[1]);
    sd.doppler_hz = atof(argv[2]);
    sd.dop_rate = atof(argv[3]);
    sd.cp_rate = atof(argv[4]);
    sd.ref_hop = atoll(argv[5]);
    const long long hop0 = atoll(argv[6]);
    const int prn = (argc > 7) ? atoi(argv[7]) : 1;

    // The TRACKER's bank and the production geometry -- same constants e2e uses.
    const double sample_rate = 3.2e9, f_offset = 1176.45e6;
    const int spectrum_length = 8192, num_taps = 4;
    const gnss::SignalDescriptor* sig = gnss::signal_by_name("GPS_L5_Q_NH");
    gnss::ChannelizedReplicaBank tbank(*sig, sample_rate, f_offset, spectrum_length, num_taps,
                                       dsp::Window::Hamming, {prn});

    const auto pr = gnss::propagate_seed(tbank, sd, hop0, sample_rate, f_offset, 0.0);
    // Full precision: the whole question is whether two implementations agree, and rounding to
    // 2 dp would hide exactly the sub-chip divergence worth knowing about.
    printf("cp        %.9f\n", pr.cp);
    printf("phase_now %.9f\n", pr.phase_now);
    printf("phase_ref %.9f\n", pr.phase_ref);
    printf("doppler   %.9f\n", pr.doppler_hz);
    return 0;
}
