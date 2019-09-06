#ifndef KOTEKAN_MODE_HPP
#define KOTEKAN_MODE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "metadata.h"
#include "restServer.hpp"

#include <map>

using std::map;
using std::string;

// doxygen wants the namespace to be documented somewhere
/*!
 *  \addtogroup kotekan
 *  @{
 */
//! Kotekan namespace
namespace kotekan {

class kotekanMode {
public:
    kotekanMode(Config& config);
    virtual ~kotekanMode();

    // Allocate memory for the stages and get the configuration.
    virtual void initalize_stages();

    // Call start on all the stages.
    void start_stages();

    // Stop all the stages.
    void stop_stages();

    // Join blocks until all stages have stopped.
    void join();

private:
    Config& config;
    bufferContainer buffer_container;

    map<string, Stage*> stages;
    map<string, struct metadataPool*> metadata_pools;
    map<string, struct Buffer*> buffers;
};

} // namespace kotekan

/*! @} End of Doxygen Groups*/

#endif /* CHIME_HPP */
