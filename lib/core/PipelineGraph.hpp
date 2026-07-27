/**
 * @file PipelineGraph.hpp
 * @brief A structured model of the running pipeline, and its graphviz rendering.
 *  - kotekan::PipelineGraph
 *  - kotekan::GraphNode
 *  - kotekan::GraphEdge
 *  - kotekan::GraphCluster
 */

#ifndef PIPELINE_GRAPH_HPP
#define PIPELINE_GRAPH_HPP

#include "json.hpp" // for json

#include <deque>    // for deque
#include <map>      // for map
#include <stddef.h> // for size_t
#include <string>   // for string
#include <vector>   // for vector

namespace kotekan {

/**
 * @brief Formats a byte count for a graph label (B / KiB / MiB / GiB / TiB).
 *
 * Sizes in a pipeline graph span bytes to hundreds of gigabytes, and the exact
 * digits never matter as much as the magnitude.
 */
std::string human_bytes(size_t bytes);

/**
 * @brief Formats a data rate for a graph label (B/s ... TB/s).
 *
 * Rates are quoted in SI units, the convention for anything on a wire, while
 * @c human_bytes() uses binary units for anything in memory.
 */
std::string human_rate(double bytes_per_second);

/**
 * @brief How busy a buffer is, for the reader who is looking for the blockage.
 */
enum class BufferState {
    Unknown, ///< nothing to say about this buffer's occupancy
    Idle,    ///< no data has ever passed through it
    Flowing, ///< holding some frames, with room for more
    Full,    ///< effectively full: whatever feeds it is being held up
};

/**
 * @brief What a node represents, which is what its colour tells the reader.
 */
enum class GraphCategory {
    Buffer,   ///< a host buffer frames pass through
    Compute,  ///< a stage doing work on the CPU
    Gpu,      ///< a stage driving a GPU, and the commands it runs
    Io,       ///< a stage moving data off (or onto) this machine
    Memory,   ///< memory private to a device, not a kotekan buffer
    Endpoint, ///< the far end of an I/O stage: a socket, a port, a file
};

/// Fill and outline colours for one category of node.
struct GraphStyle {
    const char* fill;
    const char* line;
};

/// @return The colours for @p category.
GraphStyle graph_style(GraphCategory category);

/**
 * @name The colours that are not a node category
 *
 * These are named rather than inlined because a viewer embedding the graph has
 * to be able to recognise every colour it can be handed -- choco recolours the
 * whole palette for its dark theme by matching these exact values. Changing one
 * is a change to that contract, not a local edit.
 * @{
 */
/// Label text, everywhere. Set so that text carries an explicit colour rather
/// than falling back to black, which a stylesheet cannot select on.
extern const char* const graph_ink;
/// Edges and their arrowheads.
extern const char* const graph_edge_line;
/// Outline of a config-section box.
extern const char* const graph_cluster_line;
/// Background and outline of the box drawn around one device's stages and memory.
extern const char* const graph_device_fill;
extern const char* const graph_device_line;
/// Outline of a buffer with no free frame left.
extern const char* const graph_full_line;
/** @} */

/**
 * @brief The last component of a kotekan unique name ("/gen/voltage" -> "voltage").
 *
 * The full path is what identifies a node; the leaf is what a reader needs, once
 * the enclosing box says which section it came from.
 */
std::string leaf_name(const std::string& unique_name);

/**
 * @brief Guesses what a stage does from its registered type name.
 *
 * Only the GPU stage types are named exactly; everything else is matched on
 * substrings, so a new network or file stage is categorised without having to
 * be listed here. This only picks a colour -- being wrong costs nothing but a
 * misleading shade.
 *
 * @param stage_type The `kotekan_stage` type name from the config.
 */
GraphCategory stage_category(const std::string& stage_type);

/**
 * @brief A node in the pipeline graph.
 *
 * A node is identified by @c id, which must be unique across the whole graph
 * (it is what edges refer to).  Everything else is presentation: @c label_lines
 * are plain text and are escaped and joined into a single multi-line label at
 * render time, so callers never build markup themselves.
 */
struct GraphNode {
    /// Unique identifier; edges refer to nodes by this string.
    std::string id;

    /// Label content, one plain-text line per entry.  Escaped when rendered.
    std::vector<std::string> label_lines;

    /// Id of the cluster this node belongs to; empty for a top-level node.
    std::string cluster;

    /// Graphviz node attributes (shape, style, color, ...) other than the label.
    std::map<std::string, std::string> attrs;

    /// Convenience: append a label line, ignoring empty strings.
    GraphNode& add_line(const std::string& line);

    /// Convenience: set a graphviz attribute.
    GraphNode& set_attr(const std::string& key, const std::string& value);

    /**
     * @brief Applies a category's shape and colours to this node.
     *
     * Every contributor styles its nodes through this, so a category looks the
     * same wherever in the pipeline it is drawn.
     */
    GraphNode& set_category(GraphCategory category);

    /**
     * @brief Marks how busy a buffer node is, on top of its category colours.
     *
     * A full buffer is outlined in red and a buffer nothing has ever flowed
     * through is dashed, so the place a pipeline is stuck can be found without
     * reading a single number.
     */
    GraphNode& set_buffer_state(BufferState state);
};

/**
 * @brief A directed edge between two nodes.
 *
 * @c from and @c to are node ids.  Referring to a node that is never added is a
 * programming error: it renders as an empty box because graphviz creates nodes
 * implicitly, so @c PipelineGraph::to_dot() reports such dangling edges rather
 * than silently drawing them.
 */
struct GraphEdge {
    std::string from;
    std::string to;

    /// Plain-text edge label; escaped when rendered.  May be empty.
    std::string label;

    /// Graphviz edge attributes (style, color, weight, ...) other than the label.
    std::map<std::string, std::string> attrs;

    GraphEdge& set_attr(const std::string& key, const std::string& value);
};

/**
 * @brief A box drawn around a set of nodes (a graphviz `cluster_*` subgraph).
 *
 * Clusters may nest via @c parent.  A cluster is only a visual grouping: edges
 * always connect nodes, never clusters, so a stage that renders as a cluster
 * needs an anchor node inside it for the pipeline's buffer edges to land on.
 */
struct GraphCluster {
    /// Unique identifier (the rendered subgraph is named `cluster_<id>`).
    std::string id;

    /// Plain-text cluster label; escaped when rendered.
    std::string label;

    /// Id of the enclosing cluster; empty for a top-level cluster.
    std::string parent;

    /// Graphviz subgraph attributes (style, color, ...) other than the label.
    std::map<std::string, std::string> attrs;

    GraphCluster& set_attr(const std::string& key, const std::string& value);
};

/**
 * @brief What to put in a pipeline graph.
 *
 * Carried on the graph itself, so that everything contributing to it can see
 * what was asked for without the options being threaded through every call.
 */
struct GraphOptions {
    /// Layout direction: "LR", "TB", "RL" or "BT".
    std::string rankdir = "LR";

    /// Group stages into a box per config section.
    bool cluster = true;

    /// Draw the colour key.
    bool legend = true;

    /// List the metadata pools.
    bool pools = true;

    /// Draw each GPU stage's commands and device memory.
    bool kernels = true;

    /**
     * @brief Include what the pipeline is doing right now: rates, fill levels,
     *        CPU usage, kernel timings, and the full/idle colouring.
     *
     * Turning this off leaves the structure alone, which is what to compare
     * between two runs -- live numbers differ every time the graph is fetched.
     */
    bool runtime = true;

    /// Link buffer nodes to the endpoints that can show their contents.
    bool urls = true;
};

/**
 * @class PipelineGraph
 * @brief The pipeline's buffers, stages and their connections, as a graph.
 *
 * The graph is built once (by @c kotekanMode, with stages contributing their own
 * internal detail through @c Stage::add_graph_details()) and then rendered.
 * Keeping the model separate from the rendering means the escaping, quoting and
 * statement syntax live in exactly one place, and the same model can later be
 * served in other formats.
 *
 * Node ids are arbitrary strings -- typically a stage's unique name or a buffer
 * name -- and are quoted on output, so they may contain any character.
 */
class PipelineGraph {
public:
    /**
     * @brief Adds a node, or returns the existing node with this id.
     * @param id  Unique node identifier.
     * @return A reference to the node, valid until the graph is destroyed.
     */
    GraphNode& add_node(const std::string& id);

    /**
     * @brief Adds a cluster, or returns the existing cluster with this id.
     * @param id  Unique cluster identifier.
     * @return A reference to the cluster, valid until the graph is destroyed.
     */
    GraphCluster& add_cluster(const std::string& id);

    /**
     * @brief Adds a directed edge between two node ids.
     *
     * The nodes do not have to exist yet; they must exist by the time the graph
     * is rendered.
     */
    GraphEdge& add_edge(const std::string& from, const std::string& to);

    /// @return true if a node with this id has been added.
    bool has_node(const std::string& id) const;

    /**
     * @brief The innermost cluster that contains all of @p node_ids.
     *
     * Used to place a node next to everything that touches it: a buffer whose
     * every producer and consumer live on one device belongs inside that
     * device's box, while one shared across devices belongs in the box that
     * holds them both.
     *
     * @param node_ids  Nodes that must all be inside the returned cluster.
     * @return The cluster id, or an empty string when the nodes have no common
     *         cluster (or any of them is unknown or at the top level).
     */
    std::string common_cluster(const std::vector<std::string>& node_ids) const;

    /**
     * @brief Renders the graph in graphviz `dot` format.
     *
     * Nodes are emitted inside their cluster, clusters nest, and every statement
     * is terminated and newline separated.  An edge naming a node that was never
     * added is dropped and reported in a comment, since graphviz would otherwise
     * conjure up an empty node for it.
     */
    std::string to_dot() const;

    /**
     * @brief The same graph as JSON: nodes, edges and clusters.
     *
     * For anything that wants to lay the pipeline out itself, or diff two
     * graphs, rather than parse DOT to get at the structure.
     */
    nlohmann::json to_json() const;

    /// What was asked to be included; readable by everything building the graph.
    GraphOptions options;

    /// Graph-wide attributes, applied in a `graph [...]` statement.
    std::map<std::string, std::string> graph_attrs;

    /// Default attributes for all nodes, applied in a `node [...]` statement.
    std::map<std::string, std::string> node_attrs;

    /// Default attributes for all edges, applied in an `edge [...]` statement.
    std::map<std::string, std::string> edge_attrs;

    /// Comment lines emitted at the top of the graph.
    std::vector<std::string> header_comments;

    /**
     * @brief Escapes plain text for use inside a graphviz HTML-like label.
     *
     * Node and buffer names come from the config, so they can contain characters
     * that would otherwise terminate or corrupt the label.
     */
    static std::string escape_html(const std::string& text);

    /// Quotes and escapes a string for use as a DOT id or double-quoted value.
    static std::string quote(const std::string& text);

private:
    // Deques, not vectors: callers hold on to the references handed back by
    // add_node()/add_cluster()/add_edge() while adding further elements, and a
    // deque keeps references valid across insertion at the end.

    /// Nodes in insertion order; the map indexes into @c _nodes by id.
    std::deque<GraphNode> _nodes;
    std::map<std::string, size_t> _node_index;

    std::deque<GraphCluster> _clusters;
    std::map<std::string, size_t> _cluster_index;

    std::deque<GraphEdge> _edges;

    /// Renders one node statement, indented by @c indent.
    std::string node_dot(const GraphNode& node, const std::string& indent) const;

    /// Renders the clusters whose parent is @c parent, and the nodes they hold.
    std::string cluster_dot(const std::string& parent, const std::string& indent) const;
};

} // namespace kotekan

#endif // PIPELINE_GRAPH_HPP
