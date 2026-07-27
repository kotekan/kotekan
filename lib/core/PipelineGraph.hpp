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

#include <deque>    // for deque
#include <map>      // for map
#include <stddef.h> // for size_t
#include <string>   // for string
#include <vector>   // for vector

namespace kotekan {

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
     * @brief Renders the graph in graphviz `dot` format.
     *
     * Nodes are emitted inside their cluster, clusters nest, and every statement
     * is terminated and newline separated.  An edge naming a node that was never
     * added is dropped and reported in a comment, since graphviz would otherwise
     * conjure up an empty node for it.
     */
    std::string to_dot() const;

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
