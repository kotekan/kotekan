#include "PipelineGraph.hpp"

#include "fmt.hpp" // for format

#include <algorithm> // for max, min, sort
#include <cctype>    // for tolower
#include <cmath>     // for isnan
#include <set>       // for set
#include <utility>   // for pair

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

std::string array_layout_line(const std::string& type_name,
                              const std::vector<std::ptrdiff_t>& extents,
                              const std::vector<std::string>& dimnames) {
    std::string layout;
    for (size_t d = 0; d < extents.size(); d++) {
        if (!layout.empty())
            layout += " × ";
        if (d < dimnames.size() && !dimnames[d].empty())
            layout += fmt::format("{:s}:{:d}", dimnames[d], extents[d]);
        else
            layout += fmt::format("{:d}", extents[d]);
    }
    return fmt::format("{:s} {:s}", type_name, layout);
}

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
    flag("gpu_table", options.gpu_table);

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
                                       const std::set<std::string>& cyclic,
                                       const std::set<std::string>& skip) const {
    std::string dot;
    for (const auto& cluster : _clusters) {
        if (render_parent(cluster, cyclic) != parent || skip.count(cluster.id))
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
        dot += cluster_dot(cluster.id, inner, cyclic, skip);
        dot += fmt::format("{:s}}}\n", indent);
    }
    return dot;
}

/// One command in a device view, and the rows it touches.
struct DeviceColumn {
    const GraphNode* node;
    std::vector<size_t> reads, writes;
    std::vector<std::string> from_host, to_host;
    int level = 0;
};

struct PipelineGraph::DeviceLayout {
    /// Every node in the device box, drawn or not.
    std::set<std::string> nodes;
    std::vector<const GraphNode*> rows;
    /// Which rows are drawn as full buffers.
    std::vector<bool> full;
    std::vector<DeviceColumn> columns;
};

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

    // A device box drawn as a table: its nodes are replaced by the table, and
    // edges to them by edges to the ports of the rows they belong to. A box
    // with no memory drawn (kernels off) keeps its nodes: its rings are plain
    // buffer nodes there, which the table has no place for.
    std::set<std::string> tables;
    std::map<std::string, std::pair<std::string, std::string>> table_port; // table, row
    std::string table_nodes;
    for (const auto& cluster : _clusters) {
        auto fill = cluster.attrs.find("fillcolor");
        if (!options.gpu_table || !cluster.parent.empty() || fill == cluster.attrs.end()
            || fill->second != graph_device_fill)
            continue;
        const DeviceLayout layout = device_layout(cluster.id);
        if (layout.columns.empty() || layout.rows.empty())
            continue;
        tables.insert(cluster.id);
        for (const auto& id : layout.nodes)
            table_port[id] = {cluster.id, ""};
        for (size_t c = 0; c < layout.columns.size(); c++)
            table_port[layout.columns[c].node->id] = {cluster.id, fmt::format(fmt("{:d}"), c)};
        table_nodes += fmt::format(fmt("{:s}{:s} [shape=\"plain\", label=<{:s}>];\n"), indent,
                                   quote(cluster.id), device_table(layout, cluster.label));
    }

    // Top-level nodes first, then the clusters (and the nodes they hold).
    for (const auto& node : _nodes)
        if (node.cluster.empty())
            dot += node_dot(node, indent);
    dot += table_nodes;
    dot += cluster_dot("", indent, cyclic, tables);

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
        std::string from = quote(edge.from), to = quote(edge.to);
        auto from_port = table_port.find(edge.from);
        auto to_port = table_port.find(edge.to);
        const bool from_table = from_port != table_port.end();
        const bool to_table = to_port != table_port.end();
        // An edge inside one table is drawn by it. A node replaced by a table
        // without a row of its own has only execution-order edges: drop them.
        if (from_table && to_table && from_port->second.first == to_port->second.first)
            continue;
        if ((from_table && from_port->second.second.empty())
            || (to_table && to_port->second.second.empty()))
            continue;
        if (from_table)
            from = fmt::format(fmt("{:s}:\"out{:s}\":e"), quote(from_port->second.first),
                               from_port->second.second);
        if (to_table)
            to = fmt::format(fmt("{:s}:\"in{:s}\":w"), quote(to_port->second.first),
                             to_port->second.second);
        std::string attrs = attrs_dot(edge.attrs);
        if (!edge.label.empty()) {
            std::string label = fmt::format("label=<{:s}>", escape_html(edge.label));
            attrs = attrs.empty() ? label : fmt::format("{:s}, {:s}", label, attrs);
        }
        if (attrs.empty())
            dot += fmt::format("{:s}{:s} -> {:s};\n", indent, from, to);
        else
            dot += fmt::format("{:s}{:s} -> {:s} [{:s}];\n", indent, from, to, attrs);
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

PipelineGraph::DeviceLayout PipelineGraph::device_layout(const std::string& device_cluster) const {
    DeviceLayout layout;
    std::set<std::string> device_clusters = {device_cluster};
    for (const auto& cluster : _clusters)
        if (cluster.parent == device_cluster)
            device_clusters.insert(cluster.id);

    const std::string memory_fill = graph_style(GraphCategory::Memory).fill;
    std::vector<const GraphNode*> rows;
    std::map<std::string, size_t> row_of;
    for (const auto& node : _nodes) {
        if (!device_clusters.count(node.cluster))
            continue;
        layout.nodes.insert(node.id);
        auto fill = node.attrs.find("fillcolor");
        if (fill != node.attrs.end() && fill->second == memory_fill) {
            row_of[node.id] = rows.size();
            rows.push_back(&node);
        }
    }

    // The dotted edges are execution order within a stage, not data.
    std::map<std::string, DeviceColumn> columns;
    for (const auto& edge : _edges) {
        auto style = edge.attrs.find("style");
        if (style != edge.attrs.end() && style->second == "dotted")
            continue;
        if (!has_node(edge.from) || !has_node(edge.to))
            continue;
        const bool from_device = layout.nodes.count(edge.from);
        const bool to_device = layout.nodes.count(edge.to);
        const bool from_row = row_of.count(edge.from);
        const bool to_row = row_of.count(edge.to);
        if (from_row && to_device && !to_row)
            columns[edge.to].reads.push_back(row_of[edge.from]);
        else if (to_row && from_device && !from_row)
            columns[edge.from].writes.push_back(row_of[edge.to]);
        else if (from_device && !from_row && !to_device)
            columns[edge.from].to_host.push_back(edge.to);
        else if (to_device && !to_row && !from_device)
            columns[edge.to].from_host.push_back(edge.from);
    }
    if (columns.empty())
        return layout;
    for (auto& column : columns)
        column.second.node = &_nodes[_node_index.at(column.first)];

    // A column comes after every column writing a row it reads. The passes
    // stop at the column count, so a loop through a ring cannot spin forever.
    std::map<size_t, std::vector<const DeviceColumn*>> writers;
    for (const auto& column : columns)
        for (size_t row : column.second.writes)
            writers[row].push_back(&column.second);
    for (size_t pass = 0; pass < columns.size(); pass++) {
        bool changed = false;
        for (auto& column : columns) {
            int level = 0;
            for (size_t row : column.second.reads)
                for (const DeviceColumn* writer : writers[row])
                    if (writer != &column.second)
                        level = std::max(level, writer->level + 1);
            changed |= level != column.second.level;
            column.second.level = level;
        }
        if (!changed)
            break;
    }

    std::vector<const DeviceColumn*> order;
    for (const auto& column : columns)
        order.push_back(&column.second);
    std::sort(order.begin(), order.end(), [this](const DeviceColumn* a, const DeviceColumn* b) {
        if (a->level != b->level)
            return a->level < b->level;
        if (a->node->cluster != b->node->cluster)
            return a->node->cluster < b->node->cluster;
        return _node_index.at(a->node->id) < _node_index.at(b->node->id);
    });

    // Rows in the order their first writer appears; unwritten rows first.
    std::vector<size_t> first_writer(rows.size(), 0);
    for (size_t row = 0; row < rows.size(); row++)
        for (size_t c = order.size(); c-- > 0;)
            for (size_t written : order[c]->writes)
                if (written == row)
                    first_writer[row] = c + 1;
    std::vector<size_t> row_order(rows.size());
    for (size_t row = 0; row < rows.size(); row++)
        row_order[row] = row;
    std::stable_sort(row_order.begin(), row_order.end(), [&first_writer](size_t a, size_t b) {
        return first_writer[a] < first_writer[b];
    });

    std::vector<size_t> position(rows.size());
    for (size_t k = 0; k < row_order.size(); k++) {
        const GraphNode* row = rows[row_order[k]];
        position[row_order[k]] = k;
        layout.rows.push_back(row);
        auto color = row->attrs.find("color");
        layout.full.push_back(color != row->attrs.end() && color->second == graph_full_line);
    }
    for (const DeviceColumn* column : order) {
        layout.columns.push_back(*column);
        for (size_t& row : layout.columns.back().reads)
            row = position[row];
        for (size_t& row : layout.columns.back().writes)
            row = position[row];
    }
    return layout;
}

/// Approximate width of @p text in px, for sizing the margins.
static double text_width(const std::string& text, double font_px) {
    // Count code points, not bytes: layout lines carry `×` and `·`.
    size_t chars = 0;
    for (unsigned char c : text)
        if ((c & 0xC0) != 0x80)
            chars++;
    return 0.6 * font_px * chars;
}

static std::string joined_lines(const GraphNode& node) {
    std::string out;
    for (const auto& line : node.label_lines)
        out += (out.empty() ? "" : "\n") + line;
    return out;
}

std::string PipelineGraph::column_label(const GraphNode& node) const {
    std::string label = node.label_lines.empty() ? node.id : node.label_lines[0];
    auto cluster = _cluster_index.find(node.cluster);
    if (cluster != _cluster_index.end() && !_clusters[cluster->second].label.empty())
        label = _clusters[cluster->second].label + " · " + label;
    return label;
}

std::string PipelineGraph::to_device_svg(const std::string& device_cluster) const {
    const DeviceLayout layout = device_layout(device_cluster);
    if (layout.columns.empty())
        return "";
    const std::vector<const GraphNode*>& rows = layout.rows;
    const std::vector<DeviceColumn>& order = layout.columns;

    // Column labels slope up and to the right at 55 degrees.
    const double row_h = 34, col_w = 44, band_h = 46, gauge = 170;
    std::vector<std::string> column_labels;
    double label_width = 0.0, overhang = 0.0;
    for (const DeviceColumn& column : order) {
        const std::string label = column_label(*column.node);
        column_labels.push_back(label);
        label_width = std::max(label_width, text_width(label, 10));
        // How far this label reaches past the last column.
        const double from_end = (double)column_labels.size() - 0.5 - (double)order.size();
        overhang = std::max(overhang, col_w * from_end + 0.57 * text_width(label, 10));
    }
    double left = 160.0;
    for (const GraphNode* row : rows) {
        left = std::max(
            left, 24 + text_width(row->label_lines.empty() ? row->id : row->label_lines[0], 11));
        left = std::max(left, 24 + text_width(row->label_lines.back(), 9));
    }

    const double top = 30 + 0.82 * label_width;
    const double right = std::max(gauge, overhang + 20);
    const double width = left + col_w * order.size() + right;
    const double height = top + 2 * band_h + row_h * rows.size() + 24;
    const double y_in = top + band_h / 2;
    const double y_out = top + band_h + row_h * rows.size() + band_h / 2;
    auto row_y = [&](size_t row) { return top + band_h + row_h * row + row_h / 2; };

    const std::string gpu_line = graph_style(GraphCategory::Gpu).line;
    const std::string host_fill = graph_style(GraphCategory::Buffer).fill;
    const std::string host_line = graph_style(GraphCategory::Buffer).line;
    const std::string memory_line = graph_style(GraphCategory::Memory).line;

    std::string svg = fmt::format(
        fmt("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.0f}\" height=\"{:.0f}\" "
            "font-family=\"Helvetica,Arial,sans-serif\" font-size=\"11\" fill=\"{:s}\">\n"),
        width, height, graph_ink);

    // A shaded band behind every other stage's columns.
    bool shade = false;
    for (size_t c = 0; c < order.size(); c++) {
        if (c == 0 || order[c].node->cluster != order[c - 1].node->cluster)
            shade = !shade;
        if (shade)
            svg += fmt::format(fmt("<rect x=\"{:.1f}\" y=\"{:.1f}\" width=\"{:.1f}\" "
                                   "height=\"{:.1f}\" fill=\"{:s}\"/>\n"),
                               left + col_w * c, top - 0.82 * label_width, col_w,
                               height - top + 0.82 * label_width, graph_device_fill);
    }

    for (double y : {y_in, y_out}) {
        svg += fmt::format(fmt("<rect x=\"0\" y=\"{:.1f}\" width=\"{:.1f}\" height=\"{:.1f}\" "
                               "fill=\"{:s}\"/>\n"),
                           y - band_h / 2 + 3, width, band_h - 6, host_fill);
        svg += fmt::format(fmt("<text x=\"12\" y=\"{:.1f}\" fill=\"{:s}\" "
                               "font-weight=\"bold\">{:s}</text>\n"),
                           y + 4, host_line, y == y_in ? "from host" : "to host");
    }

    for (size_t row = 0; row < rows.size(); row++) {
        const GraphNode& node = *rows[row];
        const double y = row_y(row);
        const bool full = layout.full[row];
        svg += fmt::format(fmt("<line x1=\"{:.1f}\" y1=\"{:.1f}\" x2=\"{:.1f}\" y2=\"{:.1f}\" "
                               "stroke=\"{:s}\"/>\n"),
                           left - 6, y, width - right + 6, y, graph_cluster_line);
        svg += fmt::format(fmt("<g><title>{:s}</title><text x=\"12\" y=\"{:.1f}\" fill=\"{:s}\" "
                               "font-weight=\"bold\">{:s}</text>"),
                           escape_html(joined_lines(node)), y - 2, memory_line,
                           escape_html(node.label_lines.empty() ? node.id : node.label_lines[0]));
        if (node.label_lines.size() > 1)
            svg += fmt::format(fmt("<text x=\"12\" y=\"{:.1f}\" font-size=\"9\">{:s}</text>"),
                               y + 10, escape_html(node.label_lines.back()));
        svg += "</g>\n";
        if (!std::isnan(node.occupancy)) {
            const double x = width - right + 16;
            const double fraction = std::min(1.0, std::max(0.0, node.occupancy));
            svg += fmt::format(fmt("<rect x=\"{:.1f}\" y=\"{:.1f}\" width=\"110\" height=\"10\" "
                                   "fill=\"none\" stroke=\"{:s}\"/>\n"),
                               x, y - 5, graph_edge_line);
            svg += fmt::format(fmt("<rect x=\"{:.1f}\" y=\"{:.1f}\" width=\"{:.1f}\" "
                                   "height=\"10\" fill=\"{:s}\"/>\n"),
                               x, y - 5, 110 * fraction, full ? graph_full_line : memory_line);
            svg += fmt::format(fmt("<text x=\"{:.1f}\" y=\"{:.1f}\" font-size=\"10\">"
                                   "{:s}</text>\n"),
                               x + 116, y + 4,
                               full ? std::string("full")
                                    : fmt::format(fmt("{:.0f}%"), 100 * fraction));
        }
    }

    for (size_t c = 0; c < order.size(); c++) {
        const DeviceColumn& column = order[c];
        const double x = left + col_w * c + col_w / 2;
        bool stalled = false;
        std::vector<double> ys;
        for (size_t row : column.reads)
            ys.push_back(row_y(row));
        for (size_t row : column.writes) {
            ys.push_back(row_y(row));
            stalled |= layout.full[row];
        }
        if (!column.from_host.empty())
            ys.push_back(y_in);
        if (!column.to_host.empty())
            ys.push_back(y_out);
        const std::string line = stalled ? graph_full_line : gpu_line;
        std::string tooltip = joined_lines(*column.node);
        for (const auto& host : column.from_host)
            tooltip += "\nfrom " + host;
        for (const auto& host : column.to_host)
            tooltip += "\nto " + host;
        svg += fmt::format(fmt("<g><title>{:s}</title>\n"), escape_html(tooltip));
        if (ys.size() > 1)
            svg += fmt::format(fmt("<line x1=\"{0:.1f}\" y1=\"{1:.1f}\" x2=\"{0:.1f}\" "
                                   "y2=\"{2:.1f}\" stroke=\"{3:s}\" stroke-width=\"2.5\"/>\n"),
                               x, *std::min_element(ys.begin(), ys.end()),
                               *std::max_element(ys.begin(), ys.end()), line);
        for (size_t row : column.reads)
            svg += fmt::format(fmt("<circle cx=\"{:.1f}\" cy=\"{:.1f}\" r=\"5\" fill=\"#ffffff\" "
                                   "stroke=\"{:s}\" stroke-width=\"2\"/>\n"),
                               x, row_y(row), line);
        for (size_t row : column.writes)
            svg += fmt::format(fmt("<circle cx=\"{:.1f}\" cy=\"{:.1f}\" r=\"5.5\" "
                                   "fill=\"{:s}\"/>\n"),
                               x, row_y(row), line);
        if (!column.from_host.empty())
            svg += fmt::format(fmt("<path d=\"M{:.1f} {:.1f} l-5 -9 h10 z\" fill=\"{:s}\"/>\n"), x,
                               y_in + 2, host_line);
        if (!column.to_host.empty())
            svg += fmt::format(fmt("<path d=\"M{:.1f} {:.1f} l-5 -9 h10 z\" fill=\"{:s}\"/>\n"), x,
                               y_out + 6, host_line);
        svg += fmt::format(fmt("<text transform=\"translate({:.1f},{:.1f}) rotate(-55)\" "
                               "fill=\"{:s}\" font-size=\"10\">{:s}</text></g>\n"),
                           x + 3, top - 6, line, escape_html(column_labels[c]));
    }

    svg += fmt::format(fmt("<text x=\"12\" y=\"{:.1f}\" font-size=\"10\">"
                           "○ reads · ● writes · ▼ to or from a host buffer · "
                           "shaded columns: one stage</text>\n"),
                       height - 8);
    svg += "</svg>\n";
    return svg;
}

// Sizes in a GPU table's line cells, in points: the row height, the box a
// mark sits in, and the width of the lines joining marks.
static const int table_row_height = 24, table_mark = 14, table_line = 2;

/// A fixed-size cell of a line cell's nested table, filled with @p fill unless empty.
static std::string grid_cell(int width, int height, const std::string& fill,
                             const char* span = "") {
    return fmt::format(fmt("<TD{:s} WIDTH=\"{:d}\" HEIGHT=\"{:d}\" FIXEDSIZE=\"TRUE\"{:s}></TD>"),
                       span, width, height,
                       fill.empty() ? "" : fmt::format(fmt(" BGCOLOR=\"{:s}\""), fill));
}

/**
 * One (command, region) cell of a GPU table, @p width points wide: a mark, if
 * any, and lines leaving the middle of the cell towards each side asked for.
 *
 * Graphviz tables cannot draw lines, so they are thin filled cells of a
 * nested table, sized to fill the cell exactly. A cell without a mark has a
 * line all the way across or none, so it needs fewer cells.
 */
static std::string line_cell(int width, const char* mark, const std::string& mark_ink, bool up,
                             bool down, bool left, bool right, const std::string& v_ink,
                             const std::string& h_ink) {
    const bool vertical = up || down;
    const bool horizontal = left || right;
    if (!mark && !vertical && !horizontal)
        return "<TD></TD>";

    const std::string none;
    const int pad = (table_mark - table_line) / 2;
    const int left_w = width / 2 - table_line / 2 - pad;
    const int right_w = width - left_w - table_mark;
    const int top_h = table_row_height / 2 - table_line / 2 - pad;
    const int bottom_h = table_row_height - top_h - table_mark;

    std::string cell = "<TD CELLPADDING=\"0\"><TABLE BORDER=\"0\" CELLBORDER=\"0\" "
                       "CELLSPACING=\"0\" CELLPADDING=\"0\">";
    if (mark) {
        // Columns: left_w, pad, line, pad, right_w; rows likewise. The mark
        // spans the middle three of each. The top and bottom rows are five
        // cells, not merged ones: graphviz sizes overlapping spans loosely.
        const std::string up_ink = up ? v_ink : none, down_ink = down ? v_ink : none;
        cell += "<TR>" + grid_cell(left_w, top_h, none) + grid_cell(pad, top_h, none)
                + grid_cell(table_line, top_h, up_ink) + grid_cell(pad, top_h, none)
                + grid_cell(right_w, top_h, none) + "</TR>";
        cell += "<TR>" + grid_cell(left_w, pad, none)
                + fmt::format(fmt("<TD ROWSPAN=\"3\" COLSPAN=\"3\" WIDTH=\"{0:d}\" "
                                  "HEIGHT=\"{0:d}\" FIXEDSIZE=\"TRUE\"><FONT POINT-SIZE=\"13\" "
                                  "COLOR=\"{1:s}\">{2:s}</FONT></TD>"),
                              table_mark, mark_ink, mark)
                + grid_cell(right_w, pad, none) + "</TR>";
        cell += "<TR>" + grid_cell(left_w, table_line, left ? h_ink : none)
                + grid_cell(right_w, table_line, right ? h_ink : none) + "</TR>";
        cell += "<TR>" + grid_cell(left_w, pad, none) + grid_cell(right_w, pad, none) + "</TR>";
        cell += "<TR>" + grid_cell(left_w, bottom_h, none) + grid_cell(pad, bottom_h, none)
                + grid_cell(table_line, bottom_h, down_ink) + grid_cell(pad, bottom_h, none)
                + grid_cell(right_w, bottom_h, none) + "</TR>";
    } else if (!horizontal) {
        cell += "<TR>" + grid_cell(left_w + pad, table_row_height, none)
                + grid_cell(table_line, table_row_height, v_ink)
                + grid_cell(pad + right_w, table_row_height, none) + "</TR>";
    } else {
        // The horizontal line is drawn across a vertical one: they cross,
        // they do not join.
        const std::string upright = vertical ? v_ink : none;
        cell += "<TR>" + grid_cell(left_w + pad, top_h + pad, none)
                + grid_cell(table_line, top_h + pad, upright)
                + grid_cell(pad + right_w, top_h + pad, none) + "</TR>";
        cell += "<TR>" + grid_cell(width, table_line, h_ink, " COLSPAN=\"3\"") + "</TR>";
        cell += "<TR>" + grid_cell(left_w + pad, pad + bottom_h, none)
                + grid_cell(table_line, pad + bottom_h, upright)
                + grid_cell(pad + right_w, pad + bottom_h, none) + "</TR>";
    }
    return cell + "</TABLE></TD>";
}

std::string PipelineGraph::device_table(const DeviceLayout& layout,
                                        const std::string& title) const {
    const std::string gpu_line = graph_style(GraphCategory::Gpu).line;
    const std::string memory_line = graph_style(GraphCategory::Memory).line;
    const std::string host_line = graph_style(GraphCategory::Buffer).line;
    const size_t width = layout.rows.size() + 2;

    std::string html = fmt::format(fmt("<TABLE BORDER=\"1\" COLOR=\"{:s}\" CELLBORDER=\"0\" "
                                       "CELLSPACING=\"0\" CELLPADDING=\"3\" BGCOLOR=\"{:s}\" "
                                       "ROWS=\"*\" COLUMNS=\"*\">"),
                                   graph_device_line, graph_device_fill);
    html += fmt::format(fmt("<TR><TD COLSPAN=\"{:d}\" ALIGN=\"LEFT\"><B>{:s}</B></TD></TR>"), width,
                        escape_html(title));

    // A column per row of the device view. A ring's region is named on the
    // ring's line for this device; other memory is named by its own label.
    // Each column is given a fixed width, from an estimate of its header's
    // width with some to spare: a column stretched wider than its line cells
    // would leave gaps in the lines.
    const std::string region_prefix = title + ": ";
    std::vector<int> widths;
    bool any_gauge = false;
    html += "<TR><TD></TD>";
    for (const GraphNode* row : layout.rows) {
        std::string name = row->label_lines.empty() ? row->id : row->label_lines[0];
        std::string size = row->label_lines.size() > 1 ? row->label_lines.back() : "";
        for (const auto& line : row->label_lines) {
            if (line.compare(0, region_prefix.size(), region_prefix) != 0)
                continue;
            name = line.substr(region_prefix.size());
            size.clear();
            const size_t dot = name.find(" · ");
            if (dot != std::string::npos) {
                size = name.substr(dot + std::string(" · ").size());
                name.resize(dot);
            }
        }
        const std::string suffix = "_buffer";
        if (name.size() > suffix.size()
            && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            name.resize(name.size() - suffix.size());
        const double text = std::max(1.2 * text_width(name, 9), text_width(size, 8));
        widths.push_back(2 * (int)std::ceil((std::max(text, 44.0) + 12) / 2));
        html +=
            fmt::format(fmt("<TD CELLPADDING=\"0\" WIDTH=\"{:d}\" HEIGHT=\"30\" FIXEDSIZE=\"TRUE\">"
                            "<FONT POINT-SIZE=\"9\" COLOR=\"{:s}\"><B>{:s}</B></FONT><BR/>"
                            "<FONT POINT-SIZE=\"8\" COLOR=\"{:s}\">{:s}</FONT></TD>"),
                        widths.back(), memory_line, escape_html(name), graph_edge_line,
                        escape_html(size.empty() ? " " : size));
        any_gauge |= !std::isnan(row->occupancy);
    }
    html += "<TD></TD></TR>";

    if (any_gauge) {
        html += fmt::format(fmt("<TR><TD ALIGN=\"RIGHT\"><FONT POINT-SIZE=\"8\" COLOR=\"{:s}\">"
                                "readable</FONT></TD>"),
                            graph_edge_line);
        for (size_t row = 0; row < layout.rows.size(); row++) {
            const double occupancy = layout.rows[row]->occupancy;
            if (std::isnan(occupancy)) {
                html += "<TD></TD>";
                continue;
            }
            const int bar = 40;
            const double fraction = std::min(1.0, std::max(0.0, occupancy));
            const int filled = (int)std::lround(bar * fraction);
            std::string cells;
            if (filled > 0)
                cells += fmt::format(fmt("<TD WIDTH=\"{:d}\" HEIGHT=\"7\" FIXEDSIZE=\"TRUE\" "
                                         "BGCOLOR=\"{:s}\"></TD>"),
                                     filled, layout.full[row] ? graph_full_line : memory_line);
            if (filled < bar)
                cells += fmt::format(fmt("<TD WIDTH=\"{:d}\" HEIGHT=\"7\" FIXEDSIZE=\"TRUE\">"
                                         "</TD>"),
                                     bar - filled);
            html += fmt::format(
                fmt("<TD><TABLE BORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"0\"><TR><TD>"
                    "<TABLE BORDER=\"1\" COLOR=\"{:s}\" CELLBORDER=\"0\" CELLSPACING=\"0\" "
                    "CELLPADDING=\"0\"><TR>{:s}</TR></TABLE></TD></TR><TR><TD>"
                    "<FONT POINT-SIZE=\"8\" COLOR=\"{:s}\">{:s}</FONT></TD></TR></TABLE></TD>"),
                graph_edge_line, cells, layout.full[row] ? graph_full_line : graph_edge_line,
                layout.full[row] ? std::string("full")
                                 : fmt::format(fmt("{:.0f}%"), 100 * fraction));
        }
        html += "<TD></TD></TR>";
    }

    // A row per command, marking the regions it reads and writes. A
    // horizontal line joins a command's marks, running to the table's edge
    // for a host copy; a vertical one joins a region's first and last user.
    const int regions = (int)layout.rows.size();
    std::vector<int> first_user(regions, (int)layout.columns.size()), last_user(regions, -1);
    for (size_t c = 0; c < layout.columns.size(); c++) {
        for (const auto* rows : {&layout.columns[c].reads, &layout.columns[c].writes}) {
            for (size_t row : *rows) {
                first_user[row] = std::min(first_user[row], (int)c);
                last_user[row] = std::max(last_user[row], (int)c);
            }
        }
    }
    for (size_t c = 0; c < layout.columns.size(); c++) {
        const DeviceColumn& column = layout.columns[c];
        bool stalled = false;
        for (size_t row : column.writes)
            stalled |= layout.full[row];
        const std::string ink = stalled ? graph_full_line : gpu_line;
        const std::string& command =
            column.node->label_lines.empty() ? column.node->id : column.node->label_lines[0];
        std::string stage;
        auto cluster = _cluster_index.find(column.node->cluster);
        if (cluster != _cluster_index.end())
            stage = _clusters[cluster->second].label;

        html +=
            fmt::format(fmt("<TR><TD PORT=\"in{:d}\" ALIGN=\"LEFT\"><FONT COLOR=\"{:s}\">"
                            "{:s}</FONT> <FONT POINT-SIZE=\"9\" COLOR=\"{:s}\">{:s}</FONT></TD>"),
                        c, ink, escape_html(stage), graph_edge_line, escape_html(command));
        int first = column.from_host.empty() ? regions : -1;
        int last = column.to_host.empty() ? -1 : regions;
        for (const auto* rows : {&column.reads, &column.writes}) {
            for (size_t row : *rows) {
                first = std::min(first, (int)row);
                last = std::max(last, (int)row);
            }
        }
        for (size_t row = 0; row < layout.rows.size(); row++) {
            const char* mark = nullptr;
            if (std::find(column.writes.begin(), column.writes.end(), row) != column.writes.end())
                mark = "●";
            else if (std::find(column.reads.begin(), column.reads.end(), row) != column.reads.end())
                mark = "○";
            const int r = (int)row, at = (int)c;
            html += line_cell(widths[row], mark, ink, first_user[row] < at && at <= last_user[row],
                              first_user[row] <= at && at < last_user[row], first < r && r <= last,
                              first <= r && r < last, memory_line, ink);
        }
        html += fmt::format(fmt("<TD PORT=\"out{:d}\"><FONT COLOR=\"{:s}\">{:s}</FONT></TD></TR>"),
                            c, host_line, column.to_host.empty() ? " " : "▶");
    }

    html += fmt::format(fmt("<TR><TD COLSPAN=\"{:d}\" ALIGN=\"LEFT\"><FONT POINT-SIZE=\"8\" "
                            "COLOR=\"{:s}\">○ reads · ● writes · ▶ copies to a host buffer · "
                            "lines join only at ○ or ●"
                            "</FONT></TD></TR></TABLE>"),
                        width, graph_edge_line);
    return html;
}

} // namespace kotekan
