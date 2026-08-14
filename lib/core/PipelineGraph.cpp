#include "PipelineGraph.hpp"

#include "fmt.hpp" // for format

#include <cctype>  // for tolower
#include <set>     // for set
#include <utility> // for pair

namespace kotekan {

std::string human_bytes(size_t bytes) {
    static const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double size = static_cast<double>(bytes);
    size_t unit = 0;
    while (size >= 1024.0 && unit + 1 < sizeof(units) / sizeof(units[0])) {
        size /= 1024.0;
        unit++;
    }
    if (unit == 0)
        return fmt::format("{:d} B", bytes);
    return fmt::format("{:.4g} {:s}", size, units[unit]);
}

std::string human_rate(double bytes_per_second) {
    static const char* units[] = {"B", "kB", "MB", "GB", "TB"};
    double rate = bytes_per_second;
    size_t unit = 0;
    while (rate >= 1000.0 && unit + 1 < sizeof(units) / sizeof(units[0])) {
        rate /= 1000.0;
        unit++;
    }
    return fmt::format("{:.3g} {:s}/s", rate, units[unit]);
}

// Pale fills with a saturated outline of the same hue: readable behind several
// lines of label text, and distinguishable in greyscale by outline darkness.
// Each hue is far enough from the others to survive being shrunk to fit a
// 200-node pipeline on a screen, where the fill is often all one can see.
GraphStyle graph_style(GraphCategory category) {
    switch (category) {
        case GraphCategory::Buffer:
            return {"#d6e6f7", "#2a6099"}; // blue
        case GraphCategory::Gpu:
            return {"#fbdcc0", "#c2691a"}; // orange
        case GraphCategory::Io:
            return {"#e6dcf5", "#7549a8"}; // violet
        case GraphCategory::Memory:
            return {"#fbd9ea", "#b01a6b"}; // magenta
        case GraphCategory::Endpoint:
            // Teal, not a second blue: the far end of a link is the thing most
            // often mistaken for the buffer feeding it.
            return {"#d3ece4", "#2b7a63"};
        case GraphCategory::Compute:
        default:
            return {"#f7edc0", "#94800c"}; // amber
    }
}

const char* const graph_ink = "#1c2430";
const char* const graph_edge_line = "#5c6b7a";
const char* const graph_cluster_line = "#aab4bf";
// A wash of the GPU hue rather than a grey slab: the region should read as
// this device's territory, and grey competes with the section boxes.
const char* const graph_device_fill = "#fdf4ea";
const char* const graph_device_line = "#dcbf9a";
const char* const graph_full_line = "#c0392b";

std::string leaf_name(const std::string& unique_name) {
    const size_t slash = unique_name.find_last_of('/');
    if (slash == std::string::npos || slash + 1 == unique_name.size())
        return unique_name;
    return unique_name.substr(slash + 1);
}

GraphCategory stage_category(const std::string& stage_type) {
    // The GPU meta-stages, by their registered names.
    if (stage_type == "gpuProcess" || stage_type == "cudaProcess" || stage_type == "hipProcess"
        || stage_type == "clProcess")
        return GraphCategory::Gpu;

    std::string lower;
    for (char c : stage_type)
        lower += (char)std::tolower((unsigned char)c);
    static const char* const io_patterns[] = {
        "network", "dpdk",  "uplink", "send", "recv", "socket", // off the machine
        "file",    "write", "read",   "disk", "dump", "record", // onto storage
    };
    for (const char* pattern : io_patterns)
        if (lower.find(pattern) != std::string::npos)
            return GraphCategory::Io;
    return GraphCategory::Compute;
}

GraphNode& GraphNode::set_category(GraphCategory category) {
    const GraphStyle style = graph_style(category);
    switch (category) {
        case GraphCategory::Memory:
            set_attr("shape", "oval");
            break;
        case GraphCategory::Endpoint:
            set_attr("shape", "doubleoctagon");
            break;
        default:
            // Rounded boxes hold multi-line labels without wasting the corners.
            set_attr("shape", "box");
            set_attr("style", "rounded,filled");
            break;
    }
    if (attrs.find("style") == attrs.end())
        set_attr("style", "filled");
    set_attr("fillcolor", style.fill);
    set_attr("color", style.line);
    return *this;
}

GraphNode& GraphNode::set_buffer_state(BufferState state) {
    switch (state) {
        case BufferState::Full:
            // Wherever the frames are piling up is where to start looking.
            set_attr("color", graph_full_line);
            set_attr("penwidth", "2.5");
            break;
        case BufferState::Idle:
            set_attr("style", "rounded,filled,dashed");
            break;
        case BufferState::Flowing:
        case BufferState::Unknown:
        default:
            break;
    }
    return *this;
}

GraphNode& GraphNode::add_line(const std::string& line) {
    if (!line.empty())
        label_lines.push_back(line);
    return *this;
}

GraphNode& GraphNode::set_attr(const std::string& key, const std::string& value) {
    attrs[key] = value;
    return *this;
}

GraphEdge& GraphEdge::set_attr(const std::string& key, const std::string& value) {
    attrs[key] = value;
    return *this;
}

GraphCluster& GraphCluster::set_attr(const std::string& key, const std::string& value) {
    attrs[key] = value;
    return *this;
}

GraphOptions GraphOptions::from_query(const std::map<std::string, std::string>& query) {
    GraphOptions options;

    auto flag = [&query](const std::string& name, bool& value) {
        auto arg = query.find(name);
        if (arg == query.end())
            return;
        // Accept the forms people actually type, and treat a bare `?legend` as
        // asking for it.
        const std::string& text = arg->second;
        value = text.empty() || text == "1" || text == "true" || text == "yes" || text == "on";
    };
    flag("cluster", options.cluster);
    flag("legend", options.legend);
    flag("pools", options.pools);
    flag("kernels", options.kernels);
    flag("runtime", options.runtime);
    flag("urls", options.urls);

    auto rankdir = query.find("rankdir");
    if (rankdir != query.end()
        && (rankdir->second == "LR" || rankdir->second == "TB" || rankdir->second == "RL"
            || rankdir->second == "BT"))
        options.rankdir = rankdir->second;

    return options;
}

GraphNode& PipelineGraph::add_node(const std::string& id) {
    auto it = _node_index.find(id);
    if (it != _node_index.end())
        return _nodes[it->second];
    _node_index[id] = _nodes.size();
    _nodes.push_back(GraphNode{id, {}, "", {}});
    return _nodes.back();
}

GraphCluster& PipelineGraph::add_cluster(const std::string& id) {
    auto it = _cluster_index.find(id);
    if (it != _cluster_index.end())
        return _clusters[it->second];
    _cluster_index[id] = _clusters.size();
    _clusters.push_back(GraphCluster{id, "", "", {}});
    return _clusters.back();
}

GraphEdge& PipelineGraph::add_edge(const std::string& from, const std::string& to) {
    _edges.push_back(GraphEdge{from, to, "", {}});
    return _edges.back();
}

bool PipelineGraph::has_node(const std::string& id) const {
    return _node_index.find(id) != _node_index.end();
}

std::string PipelineGraph::common_cluster(const std::vector<std::string>& node_ids) const {
    // The chain of clusters holding a node, outermost first.
    auto ancestry = [this](const std::string& node_id) {
        std::vector<std::string> chain;
        auto node = _node_index.find(node_id);
        if (node == _node_index.end())
            return chain;
        std::set<std::string> seen;
        std::string cluster = _nodes[node->second].cluster;
        while (!cluster.empty()) {
            if (!seen.insert(cluster).second) {
                // The parent chain loops (the same wiring bug to_dot() breaks
                // and reports), so no cluster meaningfully contains this node:
                // treat it as top level rather than walking the loop forever.
                chain.clear();
                return chain;
            }
            chain.insert(chain.begin(), cluster);
            auto it = _cluster_index.find(cluster);
            if (it == _cluster_index.end())
                break;
            cluster = _clusters[it->second].parent;
        }
        return chain;
    };

    bool first = true;
    std::vector<std::string> common;
    for (const auto& node_id : node_ids) {
        const std::vector<std::string> chain = ancestry(node_id);
        if (first) {
            common = chain;
            first = false;
            continue;
        }
        size_t shared = 0;
        while (shared < common.size() && shared < chain.size() && common[shared] == chain[shared])
            shared++;
        common.resize(shared);
        if (common.empty())
            break;
    }
    return common.empty() ? std::string() : common.back();
}

void PipelineGraph::apply_default_style() {
    // Pipelines are long and thin, so they read left to right; the extra
    // crossing-minimisation passes are cheap next to the cost of untangling the
    // result by eye, and newrank lines the clusters up across the whole graph.
    graph_attrs["rankdir"] = options.rankdir;
    graph_attrs["nodesep"] = "0.3";
    graph_attrs["ranksep"] = "0.9";
    graph_attrs["mclimit"] = "6";
    graph_attrs["newrank"] = "true";
    // Name every colour, including the ones graphviz would default to black.
    // An unset fontcolor emits no colour at all on the text, which leaves a
    // viewer with nothing to select on when it restyles the graph -- and black
    // text is what it is left with on a dark page.
    graph_attrs["fontcolor"] = graph_ink; // cluster labels
    node_attrs["fontname"] = "Helvetica";
    node_attrs["fontsize"] = "11";
    node_attrs["fontcolor"] = graph_ink;
    edge_attrs["fontname"] = "Helvetica";
    edge_attrs["fontsize"] = "8";
    edge_attrs["fontcolor"] = graph_ink;
    edge_attrs["color"] = graph_edge_line;
}

void PipelineGraph::add_legend() {
    auto& legend = add_cluster("__legend");
    legend.label = "legend";
    legend.set_attr("style", "rounded").set_attr("color", graph_cluster_line);
    // One rank keeps the key as a compact block instead of a band stretched
    // across the whole drawing.
    legend.set_attr("rank", "same");

    struct Entry {
        const char* id;
        GraphCategory category;
        const char* text;
    };
    static const Entry entries[] = {
        {"buffer", GraphCategory::Buffer, "buffer: name / type · metadata / layout / size / fill"},
        {"compute", GraphCategory::Compute, "stage on the CPU"},
        {"gpu", GraphCategory::Gpu, "GPU stage and its commands"},
        {"io", GraphCategory::Io, "stage moving data on or off this machine"},
        {"memory", GraphCategory::Memory, "device memory (not a kotekan buffer)"},
        {"endpoint", GraphCategory::Endpoint, "the far end: a socket, a port, a file"},
    };
    for (const auto& entry : entries) {
        auto& node = add_node(std::string("__legend/") + entry.id);
        node.add_line(entry.text);
        node.cluster = legend.id;
        node.set_category(entry.category);
    }
}

std::string PipelineGraph::escape_html(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (char c : text) {
        switch (c) {
            case '&':
                out += "&amp;";
                break;
            case '<':
                out += "&lt;";
                break;
            case '>':
                out += "&gt;";
                break;
            case '"':
                out += "&quot;";
                break;
            default:
                out += c;
        }
    }
    return out;
}

std::string PipelineGraph::quote(const std::string& text) {
    std::string out = "\"";
    for (char c : text) {
        if (c == '"' || c == '\\')
            out += '\\';
        out += c;
    }
    out += '"';
    return out;
}

/// Renders an attribute map as `key="value", key="value"`, in a stable order.
static std::string attrs_dot(const std::map<std::string, std::string>& attrs) {
    std::string out;
    for (const auto& attr : attrs) {
        if (!out.empty())
            out += ", ";
        out += fmt::format("{:s}={:s}", attr.first, PipelineGraph::quote(attr.second));
    }
    return out;
}

std::string PipelineGraph::node_dot(const GraphNode& node, const std::string& indent) const {
    std::string label;
    for (const auto& line : node.label_lines) {
        if (!label.empty())
            label += "<BR/>";
        label += escape_html(line);
    }
    // An HTML-like label is delimited by <>, so it needs no quoting; a node with
    // no label at all keeps graphviz's default (the node id).
    std::string attrs = attrs_dot(node.attrs);
    if (!label.empty())
        attrs = attrs.empty() ? fmt::format("label=<{:s}>", label)
                              : fmt::format("label=<{:s}>, {:s}", label, attrs);
    if (attrs.empty())
        return fmt::format("{:s}{:s};\n", indent, quote(node.id));
    return fmt::format("{:s}{:s} [{:s}];\n", indent, quote(node.id), attrs);
}

std::set<std::string> PipelineGraph::cyclic_clusters() const {
    std::set<std::string> cyclic;
    for (const auto& cluster : _clusters) {
        // Walk up to the top, refusing to visit a cluster twice: the second
        // visit is the cycle.
        std::set<std::string> seen;
        std::string id = cluster.id;
        while (!id.empty() && seen.insert(id).second) {
            auto it = _cluster_index.find(id);
            if (it == _cluster_index.end())
                break;
            id = _clusters[it->second].parent;
        }
        if (!id.empty() && seen.count(id))
            cyclic.insert(cluster.id);
    }
    return cyclic;
}

/// The cluster @p cluster is drawn inside, with parent cycles broken: one whose
/// parent chain loops is drawn at the top level instead, so a stage that wires
/// its clusters up wrongly loses the nesting rather than everything inside it.
static const std::string& render_parent(const GraphCluster& cluster,
                                        const std::set<std::string>& cyclic) {
    static const std::string top_level;
    return cyclic.count(cluster.id) ? top_level : cluster.parent;
}

bool PipelineGraph::cluster_is_empty(const std::string& id,
                                     const std::set<std::string>& cyclic) const {
    for (const auto& node : _nodes)
        if (node.cluster == id)
            return false;
    for (const auto& cluster : _clusters)
        if (render_parent(cluster, cyclic) == id && !cluster_is_empty(cluster.id, cyclic))
            return false;
    return true;
}

std::string PipelineGraph::cluster_dot(const std::string& parent, const std::string& indent,
                                       const std::set<std::string>& cyclic) const {
    std::string dot;
    for (const auto& cluster : _clusters) {
        if (render_parent(cluster, cyclic) != parent)
            continue;
        // A section a stage was declared in, whose stages all drew themselves
        // somewhere else, is a labelled empty rectangle: it reads as a part of
        // the pipeline holding nothing rather than as a heading with nothing
        // left under it.
        if (cluster_is_empty(cluster.id, cyclic))
            continue;
        const std::string inner = indent + "    ";
        dot += fmt::format("{:s}subgraph {:s} {{\n", indent,
                           quote(fmt::format("cluster_{:s}", cluster.id)));
        if (!cluster.label.empty())
            dot += fmt::format("{:s}label=<{:s}>;\n", inner, escape_html(cluster.label));
        for (const auto& attr : cluster.attrs)
            dot += fmt::format("{:s}{:s}={:s};\n", inner, attr.first, quote(attr.second));
        for (const auto& node : _nodes)
            if (node.cluster == cluster.id)
                dot += node_dot(node, inner);
        // Nested clusters are emitted inside their parent.
        dot += cluster_dot(cluster.id, inner, cyclic);
        dot += fmt::format("{:s}}}\n", indent);
    }
    return dot;
}

std::string PipelineGraph::to_dot() const {
    const std::string indent = "    ";
    std::string dot;

    for (const auto& comment : header_comments)
        dot += fmt::format("// {:s}\n", comment);
    dot += "digraph pipeline {\n";

    if (!graph_attrs.empty())
        dot += fmt::format("{:s}graph [{:s}];\n", indent, attrs_dot(graph_attrs));
    if (!node_attrs.empty())
        dot += fmt::format("{:s}node [{:s}];\n", indent, attrs_dot(node_attrs));
    if (!edge_attrs.empty())
        dot += fmt::format("{:s}edge [{:s}];\n", indent, attrs_dot(edge_attrs));

    // A cluster whose parent chain loops would recurse forever below, so break
    // the loops first and say which clusters were in one -- like a dangling
    // edge, it is a mistake in whatever built the graph, not in the pipeline.
    const std::set<std::string> cyclic = cyclic_clusters();

    // Top-level nodes first, then the clusters (and the nodes they hold).
    for (const auto& node : _nodes)
        if (node.cluster.empty())
            dot += node_dot(node, indent);
    dot += cluster_dot("", indent, cyclic);

    // Graphviz would materialise an unknown endpoint as an unlabelled node, which
    // silently turns a wiring bug into a plausible-looking graph; drop those edges
    // and say so instead.
    std::set<std::string> dangling;
    for (const auto& edge : _edges) {
        if (!has_node(edge.from))
            dangling.insert(edge.from);
        if (!has_node(edge.to))
            dangling.insert(edge.to);
    }
    for (const auto& edge : _edges) {
        if (!has_node(edge.from) || !has_node(edge.to))
            continue;
        std::string attrs = attrs_dot(edge.attrs);
        if (!edge.label.empty()) {
            std::string label = fmt::format("label=<{:s}>", escape_html(edge.label));
            attrs = attrs.empty() ? label : fmt::format("{:s}, {:s}", label, attrs);
        }
        if (attrs.empty())
            dot += fmt::format("{:s}{:s} -> {:s};\n", indent, quote(edge.from), quote(edge.to));
        else
            dot += fmt::format("{:s}{:s} -> {:s} [{:s}];\n", indent, quote(edge.from),
                               quote(edge.to), attrs);
    }
    for (const auto& id : dangling)
        dot += fmt::format("{:s}// dropped edge(s) referring to unknown node: {:s}\n", indent, id);
    for (const auto& id : cyclic)
        dot += fmt::format("{:s}// drawn at the top level, its parent chain loops: {:s}\n", indent,
                           id);

    dot += "}\n";
    return dot;
}

nlohmann::json PipelineGraph::to_json() const {
    nlohmann::json out;
    out["graph_attrs"] = graph_attrs;

    // Same rule as to_dot(): a parent chain that loops is a wiring bug, and
    // handing it through raw would hang any consumer that recurses on parents
    // the way cluster_dot() does. Break the loops and name the clusters.
    const std::set<std::string> cyclic = cyclic_clusters();
    if (!cyclic.empty())
        out["cyclic_clusters"] = cyclic;

    out["clusters"] = nlohmann::json::array();
    for (const auto& cluster : _clusters) {
        nlohmann::json entry;
        entry["id"] = cluster.id;
        entry["label"] = cluster.label;
        entry["parent"] = cyclic.count(cluster.id) ? "" : cluster.parent;
        out["clusters"].push_back(entry);
    }

    out["nodes"] = nlohmann::json::array();
    for (const auto& node : _nodes) {
        nlohmann::json entry;
        entry["id"] = node.id;
        // The label stays split into its lines: a consumer laying the graph out
        // itself wants the parts, not one string with breaks in it.
        entry["label_lines"] = node.label_lines;
        entry["cluster"] = node.cluster;
        entry["attrs"] = node.attrs;
        out["nodes"].push_back(entry);
    }

    out["edges"] = nlohmann::json::array();
    for (const auto& edge : _edges) {
        if (!has_node(edge.from) || !has_node(edge.to))
            continue; // same rule as to_dot(): an edge to nowhere is not a fact
        nlohmann::json entry;
        entry["from"] = edge.from;
        entry["to"] = edge.to;
        if (!edge.label.empty())
            entry["label"] = edge.label;
        entry["attrs"] = edge.attrs;
        out["edges"].push_back(entry);
    }

    return out;
}

} // namespace kotekan
