#ifndef KOTEKAN_MODE_HPP
#define KOTEKAN_MODE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "bufferContainer.hpp" // for bufferContainer
#include "metadata.h"          // for metadataPool  // IWYU pragma: keep
#include "restServer.hpp"

#include <map>    // for map
#include <string> // for string


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

    // HTTP callback that dumps the current buffer state in JSON.
    void buffer_data_callback(connectionInstance& conn);

    // HTTP callback that dumps the current pipeline graph in `dot` format.
    void pipeline_dot_graph_callback(connectionInstance& conn);

private:
    Config& config;
    bufferContainer buffer_container;

    std::map<std::string, Stage*> stages;
    std::map<std::string, struct metadataPool*> metadata_pools;
    std::map<std::string, struct Buffer*> buffers;
};

struct CpuStat {
    double utime_usage = 0;
    double stime_usage = 0;
    uint32_t prev_utime = 0;
    uint32_t prev_stime = 0;
};

// List of CPU usage data <stage_name, CPU_stat>
std::map<std::string, CpuStat> ult_list;

uint32_t prev_cpu_time;

class CpuMonitor {
public:
    CpuMonitor();

    void start();

    void cpu_ult_call_back(connectionInstance& conn);

private:
    static void* track_cpu(void *);
};


} // namespace kotekan

/*! @} End of Doxygen Groups*/

#endif /* CHIME_HPP */
