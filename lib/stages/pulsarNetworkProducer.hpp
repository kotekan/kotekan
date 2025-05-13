#ifndef PULSAR_NETWORK_PRODUCER_H
#define PULSAR_NETWORK_PRODUCER_H
#include "Config.hpp" // for Config
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp" // for bufferContainer

#include <string> // for string

/**
 * @class PulsarNetworkProducer
 * @brief An example producer stage that sets each element of a buffer to a constant value.
 *
 * @par Buffers
 * @buffer out_buf The buffer to process the contents of.
 *      @buffer_format any
 *      @buffer_metadata any
 *
 * @conf    init_value  Default 0.  The value to set each element to.
 */

class PulsarNetworkProducer : public kotekan::Stage {
public:
    /**
     * @brief Constructor for the stage
     *   Note: you can use the macro STAGE_CONSTRUCTOR(PulsarNetworkProducer)
     *   if your constructor does not need additional customisation
     *   and you wish to hide the complexity.
     */
    PulsarNetworkProducer(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);

    /**
     * @brief Deconstructor - Called on shutdown, after @c main_thread has exited.
     */
    virtual ~PulsarNetworkProducer();

    /**
     * @brief Framework managed pthread.
     */
    void main_thread() override;

private:
    // Output buffer
    Buffer* out_buf;
};

#endif /* PULSAR_NETWORK_PRODUCER_H */
