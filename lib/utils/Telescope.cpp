#include "Telescope.hpp"

#include "chimeMetadata.h"

Telescope::Telescope(const std::string& log_level) {
    set_log_level(log_level);
    set_log_prefix("/telescope");
}

const Telescope& Telescope::instance() {
    if (tel_instance == nullptr) {
        FATAL_ERROR_NON_OO("Telescope singleton must be configured before use.");
    }

    return *tel_instance;
}

const Telescope& Telescope::instance(const kotekan::Config& config) {

    auto telescope_name = config.get_default<std::string>("/telescope", "name", "chime");
    if (!FACTORY(Telescope)::exists(telescope_name)) {
        FATAL_ERROR_NON_OO("Requested telescope type {} is not registered", telescope_name);
    }

    tel_instance = FACTORY(Telescope)::create_unique(telescope_name, config, "/telescope");

    return *tel_instance;
}


freq_id_t Telescope::to_freq_id(stream_t stream) const {
    if (num_freq_per_stream() != 1) {
        throw std::invalid_argument(
            "Cannot use the to_freq_id(stream) call on a multi-frequency stream.");
    }
    return to_freq_id(stream, 0);
}

freq_id_t Telescope::to_freq_id(const Buffer* buf, int ID) const {
    return to_freq_id(get_stream_id(buf, ID));
}

freq_id_t Telescope::to_freq_id(const Buffer* buf, int ID, uint32_t ind) const {
    return to_freq_id(get_stream_id(buf, ID), ind);
}

timespec Telescope::seq_length() const {
    auto dt_ns = seq_length_nsec();
    return {(time_t)(dt_ns / 1000000000), (long)(dt_ns % 1000000000)};
}