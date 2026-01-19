#ifndef BKCHECK_HPP
#define BKCHECK_HPP

#include "Config.hpp"
#include "Stage.hpp" // for Stage
#include "bufferContainer.hpp"

#include <stdint.h>    // for int32_t, uint32_t
#include <string>      // for string
#include <sys/types.h> // for uint

class bkCheck : public kotekan::Stage {
public:
    /// Constructor, also initializes internal variables from config.
    bkCheck(kotekan::Config& config, const std::string& unique_name,
                kotekan::bufferContainer& buffer_container);
    /// Destructor, cleans up local allocs.
    ~bkCheck();

    /// Primary loop to wait for buffers, verify, lather, rinse and repeat.
    void main_thread() override;


private:
    Buffer* _input_buf;
    Buffer* output_buf;

};

#endif

