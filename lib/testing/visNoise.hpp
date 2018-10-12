/*****************************************
@file
@brief Processe that adds noise to the
       visibility data.
- visNoise : public KotekanProcess
*****************************************/
#ifndef VISNOISE_H
#define VISNOISE_H

#include <random>

#include "KotekanProcess.hpp"

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
class visNoise : public KotekanProcess {
public:

    // Default constructor
    visNoise(Config &config,
             const string& unique_name,
             bufferContainer &buffer_container);

    // Main loop for the process
    void main_thread() override;

    void apply_config(uint64_t fpga_seq) override;

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

#endif /* VISNOISE_H */

