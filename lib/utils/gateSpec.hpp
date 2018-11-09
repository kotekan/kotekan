/*****************************************
@file
@brief Classes that define how to gate during accumulation.
*****************************************/
#ifndef GATE_SPEC_HPP
#define GATE_SPEC_HPP

#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <time.h>

#include "json.hpp"

#include "factory.hpp"
#include "buffer.h"
#include "visUtil.hpp"
#include "pulsarTiming.hpp"


/**
 * @brief Base class for specifying a gated data accumulation.
 *
 **/
class gateSpec {
public:

    /**
     * @brief Create a new gateSpec
     *
     * @param  name  Name of the gated dataset.
     **/
    gateSpec(const std::string& name);
    virtual ~gateSpec() = 0;

    /**
     * @brief Create a subtype by name.
     *
     * @param type Name of gateSpec subtype.
     * @param name Name of the gated dataset.
     *
     * @returns A pointer to the gateSpec instance.
     **/
    static std::unique_ptr<gateSpec> create(const std::string& type,
                                            const::std::string& name);


    /**
     * @brief A callback to update the gating specification.
     **/
    virtual bool update_spec(nlohmann::json &json) = 0;

    /**
     * @brief Get a function/closure to calculate the weights for a subsample.
     *
     * @note This must return a closure that captures by value such that its
     *       lifetime can be longer than the gateSpec object that generated it.
     *
     * @returns A function to calculate the weights.
     **/
    virtual std::function<float(timespec, timespec, float)> weight_function() const = 0;

    /**
     * @brief Is this enabled at the moment?
     **/
    const bool& enabled() const { return _enabled; }

    /**
     * @brief Get the name of the gated dataset.
     **/
    const std::string& name() const { return _name; }

protected:

    // Name of the gated dataset in the config
    const std::string _name;

    // Is the dataset enabled?
    bool _enabled = false;
};

// Create a factory for gateSpecs
CREATE_FACTORY(gateSpec, const std::string&);
#define REGISTER_GATESPEC(specType, name) REGISTER_NAMED_TYPE_WITH_FACTORY(gateSpec, specType, name)


/**
 * @brief Pulsar gating.
 *
 * Config message must contain:
 * @conf  enabled      Bool. Is the gating enabled or not.
 * @conf  pulsar_name  String. Name of the pulsar.
 * @conf  dm           Float. Dispersion measure in pc/cm^3.
 * @conf  t_ref        Float. Reference time for solution. Should be close to
 *                     the observing time.
 * @conf  phase_ref    Float. Phase of pulsar at t_ref.
 * @conf  rot_freq     Float. Rotational frequency in Hz.
 * @conf  pulse_width  Float. Width of pulse in s.
 * @conf  coeff        Array of floats. Polyco coefficients for timing solution.
 **/
class pulsarSpec : public gateSpec {

public:
    pulsarSpec(const std::string& name) : gateSpec(name) {};

    bool update_spec(nlohmann::json &json) override;
    std::function<float(timespec, timespec, float)> weight_function() const override;

private:
    // Config parameters for pulsar gating
    std::string pulsar_name;
    float dm;           // in pc / cm^3
    double tmid;        // in days since MJD
    double phase_ref;   // in number of rotations
    double rot_freq;    // in Hz
    float pulse_width;  // in s
    Polyco polyco;
};


/**
 * @brief Uniformly weight all data.
 *
 * @note This is used to implement the nominal visibility dataset. It does not
 *       actually gate.
 **/
class uniformSpec : public gateSpec {
public:
    uniformSpec(const std::string& name);
    bool update_spec(nlohmann::json &json) override;
    std::function<float(timespec, timespec, float)> weight_function() const override;
};

#endif