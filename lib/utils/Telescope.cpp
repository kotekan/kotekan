#include "Telescope.hpp"

#include "fmt.hpp" // for compile_string_to_view

#include <stdexcept> // for invalid_argument


Telescope::Telescope(const std::string& log_level) {
    set_log_level(log_level);
    set_log_prefix("/telescope");
}

const Telescope& Telescope::instance() {
    if (!tel_instance()) {
        FATAL_ERROR_NON_OO("Telescope singleton must be configured before use.");
    }

    return *tel_instance();
}

const Telescope& Telescope::instance(const kotekan::Config& config) {

    auto telescope_name = config.get_default<std::string>("/telescope", "name", "ICETelescope");
#if !defined(WITH_TESTS)
    if (telescope_name == "fake")
        WARN_NON_OO("To use the fake telescope, build with -DWITH_TESTS=ON");
#endif
    if (!FACTORY(Telescope)::exists(telescope_name)) {
        FATAL_ERROR_NON_OO("Requested telescope type {} is not registered", telescope_name);
    }

    tel_instance() = FACTORY(Telescope)::create_unique(telescope_name, config, "/telescope");

    return *tel_instance();
}

std::unique_ptr<Telescope>& Telescope::tel_instance() {
    // this must be declare in a function to ensure correct order of
    // desctructors when unwinding the ctor stack
    static std::unique_ptr<Telescope> the_tel_instance;

    return the_tel_instance;
}

freq_id_t Telescope::to_freq_id(stream_t stream) const {
    if (num_freq_per_stream() != 1) {
        throw std::invalid_argument(
            "Cannot use the to_freq_id(stream) call on a multi-frequency stream.");
    }
    return to_freq_id(stream, 0);
}

timespec Telescope::seq_length() const {
    auto dt_ns = seq_length_nsec();
    return {(time_t)(dt_ns / 1000000000), (long)(dt_ns % 1000000000)};
}
