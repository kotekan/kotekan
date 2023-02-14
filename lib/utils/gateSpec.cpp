#include "gateSpec.hpp"

#include "kotekanLogging.hpp" // for WARN, logLevel, INFO
#include "visUtil.hpp"        // for operator-, ts_to_double

#include <algorithm> // for copy
#include <exception> // for exception
#include <time.h>    // for timespec
#include <utility>   // for move


REGISTER_GATESPEC(pulsarSpec, "pulsar");
REGISTER_GATESPEC(uniformSpec, "uniform");


gateSpec::gateSpec(const std::string& name, const kotekan::logLevel log_level) :
    _name(name) {
    set_log_level(log_level);
}


std::unique_ptr<gateSpec> gateSpec::create(const std::string& type, const std::string& name,
                                           const kotekan::logLevel&& log_level) {
    return FACTORY(gateSpec)::create_unique(type, name, std::move(log_level));
}


gateSpec::~gateSpec() {}


bool pulsarSpec::update_spec(nlohmann::json& json) {

    try {
        _enabled = json.at("enabled").get<bool>();
    } catch (std::exception& e) {
        WARN("Failure reading 'enabled' from update: {:s}", e.what());
        return false;
    }

    if (!enabled()) {
        INFO("Disabling gated dataset {:s}", name());
        return true;
    }

    try {
        // Get gating specifications from config
        _pulsar_name = json.at("pulsar_name").get<std::string>();
        _coeff = json.at("coeff").get<std::vector<std::vector<float>>>();
        _dm = json.at("dm").get<float>();
        _tmid = json.at("t_ref").get<std::vector<double>>();
        _phase_ref = json.at("phase_ref").get<std::vector<double>>();
        _rot_freq = json.at("rot_freq").get<double>();
        _seg = json.at("segment").get<float>();
        _pulse_width = json.at("pulse_width").get<float>();
    } catch (std::exception& e) {
        WARN("Failure reading pulsar parameters from update: {:s}", e.what());
        return false;
    }
    try {
        _polycos = SegmentedPolyco(_rot_freq, _dm, _seg, _tmid, _phase_ref, _coeff);
    } catch (std::exception& e) {
        WARN("Could not generate polyco from config parameters: {:s}", e.what());
        return false;
    }
    INFO("Dataset {:s} now gating on pulsar {:s}", name(), _pulsar_name);

    return true;
}


std::function<float(timespec, timespec, float)> pulsarSpec::weight_function(timespec t) const {

    Polyco a_polyco;
    try {
        a_polyco = _polycos.get_polyco(t);
    } catch (std::exception& e) {
        WARN("Could not find a polyco solution for this time."
             "Will use last polyco, but timing may be off.");
        a_polyco = _polycos.get_polyco(-1);
    }
    // capture the variables needed to calculate timing
    return
        [p = a_polyco, f0 = _rot_freq, pw = _pulse_width](timespec t_s, timespec t_e, float freq) {
            // Calculate nearest pulse times of arrival
            double toa = p.next_toa(t_s, freq);
            double last_toa = toa - 1. / f0;

            // width of frame
            double fw = ts_to_double(t_e - t_s);

            // Weights are on/off for now
            if (toa < fw || last_toa + pw > 0) {
                return 1.;
            } else {
                return 0.;
            }
        };
}


nlohmann::json pulsarSpec::to_dm_json() const {
    return {{"pulsar_name", _pulsar_name},
            {"dm", _dm},
            {"t_ref", _tmid},
            {"phase_ref", _phase_ref},
            {"rot_freq", _rot_freq},
            {"segment", _seg},
            {"pulse_width", _pulse_width},
            {"coeff", _coeff}};
}


uniformSpec::uniformSpec(const std::string& name, const kotekan::logLevel log_level) :
    gateSpec(name, log_level) {
    _enabled = true;
}


bool uniformSpec::update_spec(nlohmann::json& json) {
    // Parameter not used in this spec, suppress warning.
    (void)json;

    return true;
}


std::function<float(timespec, timespec, float)> uniformSpec::weight_function(timespec t) const {
    (void)t;
    return [](timespec ts, timespec te, float freq) -> float {
        // Parameters not used in this spec, suppress warnings.
        (void)ts;
        (void)te;
        (void)freq;

        return 1.0;
    };
}
