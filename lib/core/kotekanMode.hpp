#ifndef KOTEKAN_MODE_HPP
#define KOTEKAN_MODE_HPP

#include "Config.hpp"          // for Config
#include "PipelineGraph.hpp"   // for PipelineGraph
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer, GenericBuffer
#include "bufferContainer.hpp" // for bufferContainer
#include "metadata.hpp"        // for metadataPool
#include "restServer.hpp"      // for connectionInstance
#if !defined(MAC_OSX)
#include "cpuMonitor.hpp" // for CpuMonitor
#endif

#include "json.hpp" // for json

#include <map>    // for map
#include <memory> // for shared_ptr
#include <string> // for string
#include <vector> // for vector

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

    /// Allocate memory for the stages and get the configuration.
    virtual void initalize_stages();

    /// Call start on all the stages.
    void start_stages();

    /// Stop all the stages.
    void stop_stages();

    /// Join blocks until all stages have stopped.
    void join();

    /// HTTP callback that dumps the current buffer state in JSON.
    void buffer_data_callback(connectionInstance& conn);

    /**
     * @brief Generate a json structure with active buffer data
     *
     * This json also contains all the consumer and producer data needed to generate
     * a pipeline graph.
     *
     * @return Returns JSON formatted data with all the current buffer information
     */
    nlohmann::json get_buffer_json();

    /**
     * @brief Builds the graph of the running pipeline: the buffers, the stages,
     *        and the producer/consumer relations connecting them.
     *
     * Stages contribute their own internal detail through
     * @c Stage::add_graph_details(); everything common lives here so that the
     * graph is assembled in one place and can be rendered in any format.
     *
     * @param options What to include in the graph.
     * @return The pipeline graph, as of the moment of the call.
     */
    PipelineGraph get_pipeline_graph(const GraphOptions& options = GraphOptions());

    /// HTTP callback that dumps the current pipeline graph in `dot` format.
    void pipeline_dot_graph_callback(connectionInstance& conn);

    /// HTTP callback that dumps the same graph as JSON, for clients that would
    /// rather lay it out (or diff it) themselves than parse DOT.
    void pipeline_json_graph_callback(connectionInstance& conn);

    /**
     * @brief HTTP callback serving a copy of the newest full frame in a buffer.
     *
     * Registered at `GET /buffer_frame`, which names its buffer in the required
     * `name` query parameter. Replies with a JSON object containing the frame's
     * metadata, the buffer's frame descriptor (when attached), and the leading
     * frame bytes base64 encoded under `data`. The optional `len` query
     * parameter caps the number of data bytes included, up to the frame size;
     * `len=0` returns metadata only, and no `len` returns @c default_peek_len
     * bytes.
     */
    void buffer_frame_callback(connectionInstance& conn);

    /**
     * @brief How many bytes of frame data `/buffer_frame` copies when the
     *        request does not ask for a length.
     *
     * The copy is made under the buffer lock, so an unbounded default would let
     * one request hold up every stage on a buffer for as long as it takes to
     * memcpy a frame -- hundreds of megabytes, on the pipelines this matters
     * for. Enough to see what the data looks like; a caller wanting a whole
     * frame asks for it.
     */
    static constexpr size_t default_peek_len = 64 * 1024;

private:
    Config& config;
    bufferContainer buffer_container;
#if !defined(MAC_OSX)
    CpuMonitor cpu_monitor;
#endif

    std::map<std::string, Stage*> stages;
    std::map<std::string, std::shared_ptr<metadataPool>> metadata_pools;

    std::map<std::string, GenericBuffer*> buffers;
};

} // namespace kotekan

/*! @} End of Doxygen Groups*/

#endif /* CHIME_HPP */
