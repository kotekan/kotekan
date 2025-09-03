/*****************************************
@file
@brief Stage that adds noise to the
       visibility data.
- visNoise : public Stage
*****************************************/
#ifndef VISNOISE_HPP
#define VISNOISE_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <random>   // for default_random_engine
#include <stddef.h> // for size_t
#include <string>   // for string

/**
 * @brief Adds gaussian noise.
 *
 * Add normally distributed random noise to real and imaginary parts.
 * The same distribution is used to set the weights. Note that the seed for the
 * generator is not random.
 * @conf  standard_deviation           The std dev of the noise distribution.
 * @conf  num_ev                       Number of eigenvectors in the data.
 * @conf  num_elements                 Number of elements in the data.
 * @conf  random                       If false, the noise generation will not be
 *                                     initialized with a random seed.
 **/
class visNoise : public kotekan::Stage {
public:
    // Default constructor
    visNoise(kotekan::Config& config, const std::string& unique_name,
             kotekan::bufferContainer& buffer_container);

    // Main loop for the stage
    void main_thread() override;

private:
    // random number generation
    std::default_random_engine gen;

    // config parameters
    float _standard_deviation;
    size_t _num_elements;
    size_t _num_eigenvectors;

    // Buffers
    Buffer* buf_out;
    Buffer* buf_in;
};

#endif /* VISNOISE_HPP */
