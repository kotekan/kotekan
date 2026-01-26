#include "Telescope.hpp"

#include "configUpdater.hpp"  // for configUpdater
#include "fmt.hpp" // for compile_string_to_view
#include "restServer.hpp"   // for restServer, connectionInstance
#include "timeUtil.hpp"

#include <mutex>
#include <shared_mutex>
#include <stdexcept> // for invalid_argument

using kotekan::connectionInstance;
using kotekan::restServer;

Telescope::Telescope(const std::string& tel_path, const std::string& log_level, const std::string& updatable_config_path) :
    _unique_name(tel_path) {
    set_log_level(log_level);
    set_log_prefix("/telescope");
    
    INFO("Building Telescope");

    // Set up callbacks for updating EOP and sending time0_ns
    using namespace std::placeholders;

    // Subscribe to config updates if updatable_config is set.
    if (updatable_config_path.length() > 0) {
        kotekan::configUpdater::instance().subscribe(
            updatable_config_path,
            std::bind(&Telescope::receive_eop_updates, this, _1));
    }

    INFO("Adding telescope REST GET endpoints");
    restServer& rest_server = restServer::instance();

    rest_server.register_get_callback(tel_path + "/time0_ns",
                                      std::bind(&Telescope::send_time0_ns, this, _1));
    rest_server.register_get_callback(tel_path + "/eop_table",
                                      std::bind(&Telescope::send_eop_table, this, _1));
}

Telescope::~Telescope() {
    // Must manually remove the GET callbacks
    restServer& rest_server = restServer::instance();
    rest_server.remove_get_callback(_unique_name + "/time0_ns");
    rest_server.remove_get_callback(_unique_name + "/eop_table");
    INFO_NON_OO("/telescope: removed REST GET endpoints");
}

const Telescope& Telescope::instance() {
    if (!tel_instance()) {
        FATAL_ERROR_NON_OO("Telescope singleton must be configured before use.");
    }

    return *tel_instance();
}

const Telescope& Telescope::instance(const kotekan::Config& config) {

    // This defaults to ICETelescope because the "Telescope" is virtual
    // and is not registered with the Factory.
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

bool Telescope::receive_eop_updates(nlohmann::json& json) {
    try {
        // Fill a temporary table with the updated values.
        std::vector<EOP> tmp_eop_table;
        for (const auto& elem : json.at("earth_orientation_parameter_table")) {
            INFO("Telescope EOP update: {:s}", elem.dump());
            int64_t t_ns = elem.at("time_inst_ns").get<int64_t>();
            double dut1 = elem.at("delta_UT1_inst").get<double>();
            double x_pm = elem.at("x_pm").get<double>();
            double y_pm = elem.at("y_pm").get<double>();
            tmp_eop_table.push_back(build_EOP_from_update(t_ns, dut1, x_pm, y_pm));
        }

        if (tmp_eop_table.empty()) {
            ERROR(
                "Telescope {}: earth_orientation_parameter_table update contained no entries.",
                _unique_name);
            return false;
        }

        // Sort chronologically
        std::sort(tmp_eop_table.begin(), tmp_eop_table.end(), EOP_comp_time);

        // Replace old table with new.
        {
            // Make sure no one is using the EOP table while we're updating it.
            std::unique_lock lock(_eop_lock);
            _eop_table = tmp_eop_table;
            INFO("Updated EOP Table with {:d} entries", _eop_table.size());
        }

    } catch (std::exception& e) {
        WARN("Telescope failed to read EOP update: {:s}", e.what());
        return false;
    }

    return true;
}

void Telescope::send_eop_table(connectionInstance& conn) {
    nlohmann::json reply;
    {
        std::shared_lock lock(_eop_lock);
        reply["eop_table"] = _eop_table;
    }
    conn.send_json_reply(reply);
}

void Telescope::send_time0_ns(connectionInstance& conn) {
    nlohmann::json reply;
    reply["time0_ns"] = to_time_ns(0);
    conn.send_json_reply(reply);
}

EOP Telescope::build_EOP_from_update(int64_t time_ns, double delta_ut1_inst,
                                     double xp_as, double yp_as) const {

    struct timespec ts_inst = nanosec_i64_to_timespec(time_ns);
    int64_t ut1 = get_UT1_from_time(ts_inst, delta_ut1_inst);
    double era = get_ERA_from_UT1(ut1, nullptr);

    EOP eop{.t_inst = time_ns,
            .t_ut1 = ut1,
            .delta_UT1_inst = delta_ut1_inst,
            .ERA_deg = era,
            .xp_as = xp_as,
            .yp_as = yp_as};

    return eop;
}
