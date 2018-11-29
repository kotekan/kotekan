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
 * Subclasses of need to implement weight_function to return a function which
 * given the start and end times of a frame of its frame, as well as it's
 * frequency, produces a weight with which it should be accumulated.
 *
 * @author Richard Shaw, Tristan Pinsonneault-Marotte
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
     * @param  type  Name of gateSpec subtype.
     * @param  name  Name of the gated dataset.
     *
     * @returns      A pointer to the gateSpec instance.
     **/
    static std::unique_ptr<gateSpec> create(const std::string& type,
                                            const::std::string& name);


    /**
     * @brief A callback to update the gating specification.
     *
     * @param  json  A json object with the updated config.
     *
     * @return       Did the config apply successfully.
     **/
    virtual bool update_spec(nlohmann::json &json) = 0;

    /**
     * @brief Get a function/closure to calculate the weights for a subsample.
     *
     * @param  timespec  The time the weight function is to be updated for.
     *
     * @note This must return a closure that captures by value such that its
     *       lifetime can be longer than the gateSpec object that generated it.
     *
     * @return  A function to calculate the weights.
     **/
    virtual std::function<float(timespec, timespec, float)> weight_function(timespec t) const = 0;

    /**
     * @brief Is this enabled at the moment?
     *
     * @return True if gating for this spec is enabled.
     **/
    const bool& enabled() const { return _enabled; }

    /**
     * @brief Get the name of the gated dataset.
     *
     * @return Name of the gated dataset.
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
 * @conf  rot_freq     Float. Rotational frequency in Hz.
 * @conf  pulse_width  Float. Width of pulse in s.
 * @conf  segment      Float. Length of polyco segments in s.
 * @conf  t_ref        Array of floats. Reference times (MJD) for solution
 *                     segment. Should be close to the observing time.
 * @conf  phase_ref    Array of floats. Phases of pulsar at t_ref.
 * @conf  coeff        Array of array of floats. Polyco coefficients
 *                     for every timing solution segment.
 **/
class pulsarSpec : public gateSpec {

public:

    /**
     * @brief Create a pulsar spec.
     **/
    pulsarSpec(const std::string& name) : gateSpec(name) {};

    /**
     * @brief Update the gating from a json message.
     **/
    bool update_spec(nlohmann::json &json) override;

    /**
     * @brief Return a closure te calculate the weigths.
     **/
    std::function<float(timespec, timespec, float)> weight_function(timespec t) const override;

private:
    // Config parameters for pulsar gating
    std::string _pulsar_name;
    float _dm;           // in pc / cm^3
    double _rot_freq;    // in Hz
    float _pulse_width;  // in s
    float _seg;  // length of polyco segments in s
    std::vector<double> _tmid;  // in MJD
    std::vector<double> _phase_ref;  // in number of rotations
    SegmentedPolyco _polycos;
};


/**
 * @brief Uniformly weight all data.
 *
 * @note This is used to implement the nominal visibility dataset. It does not
 *       actually gate.
 **/
class uniformSpec : public gateSpec {
public:

    /**
     * @brief Create a uniform weighted dataset.
     **/
    uniformSpec(const std::string& name);

    /**
     * @brief Update from json config. Has no effect.
     **/
    bool update_spec(nlohmann::json &json) override;

    /**
     * @brief Return the weight calculation function.
     **/
    std::function<float(timespec, timespec, float)> weight_function(timespec t) const override;
};

#endif