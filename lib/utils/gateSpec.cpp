#include "gateSpec.hpp"


REGISTER_GATESPEC(pulsarSpec, "pulsar");
REGISTER_GATESPEC(uniformSpec, "uniform");


gateSpec::gateSpec(const std::string& name) : _name(name)
{

}


std::unique_ptr<gateSpec> gateSpec::create(const std::string& type,
                                           const std::string& name)
{
    return FACTORY(gateSpec)::create_unique(type, name);
}


gateSpec::~gateSpec()
{

}


bool pulsarSpec::update_spec(nlohmann::json& json) {

    try {
        _enabled = json.at("enabled").get<bool>();
    } catch (std::exception& e) {
        WARN("Failure reading 'enabled' from update: %s", e.what());
        return false;
    }

    if (!enabled()) {
        INFO("Disabling gated dataset %s", name().c_str());
        return true;
    }

    std::vector<float> coeff;
    try {
        // Get gating specifications from config
        pulsar_name = json.at("pulsar_name").get<std::string>();
        coeff = json.at("coeff").get<std::vector<float>>();
        dm = json.at("dm").get<float>();
        tmid = json.at("t_ref").get<double>();
        phase_ref = json.at("phase_ref").get<double>();
        rot_freq = json.at("rot_freq").get<double>();
        pulse_width = json.at("pulse_width").get<float>();
    } catch (std::exception& e) {
        WARN("Failure reading pulsar parameters from update: %s", e.what());
        return false;
    }
    polyco = Polyco(tmid, dm, phase_ref, rot_freq, coeff);
    INFO("Dataset %s now gating on pulsar %s",
         name().c_str(), pulsar_name.c_str());

    return true;
}


std::function<float(timespec, timespec, float)> pulsarSpec::weight_function() const {

    // capture the variables needed to calculate timing
    return [
        p = polyco, f0 = rot_freq, pw = pulse_width
    ](timespec t_s, timespec t_e, float freq) {
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


uniformSpec::uniformSpec(const std::string& name) : gateSpec(name)
{
    _enabled = true;
}


bool uniformSpec::update_spec(nlohmann::json &json)
{
    return true;
}


std::function<float(timespec, timespec, float)> uniformSpec::weight_function() const
{
    return [](timespec ts, timespec te, float freq) -> float { return 1.0; };
}