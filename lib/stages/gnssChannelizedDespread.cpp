#include "gnssChannelizedDespread.hpp"

#include <stdexcept> // for invalid_argument

namespace gnss {

DespreadResult channelized_despread(const std::vector<std::vector<std::complex<float>>>& data_ch,
                                    const std::vector<std::vector<std::complex<float>>>& repl_ch) {
    if (data_ch.size() != repl_ch.size())
        throw std::invalid_argument("channelized_despread: channel count mismatch");

    DespreadResult r;
    r.per_channel.resize(data_ch.size());
    std::complex<double> G(0.0, 0.0);
    double energy = 0.0;

    for (size_t c = 0; c < data_ch.size(); ++c) {
        const auto& x = data_ch[c];
        const auto& rep = repl_ch[c];
        if (x.size() != rep.size())
            throw std::invalid_argument("channelized_despread: channel length mismatch");

        std::complex<double> g(0.0, 0.0);
        double e = 0.0;
        for (size_t m = 0; m < x.size(); ++m) {
            g += std::complex<double>(x[m]) * std::conj(std::complex<double>(rep[m]));
            e += std::norm(std::complex<double>(rep[m]));
        }
        r.per_channel[c] = g;
        G += g;
        energy += e;
    }

    r.correlation = G;
    r.replica_energy = energy;
    r.amplitude = (energy > 0.0) ? G / energy : std::complex<double>(0.0, 0.0);
    return r;
}

} // namespace gnss
