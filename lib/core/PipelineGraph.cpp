#include "PipelineGraph.hpp"

#include "fmt.hpp" // for format

#include <set>     // for set
#include <utility> // for pair

namespace kotekan {

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

std::string PipelineGraph::cluster_dot(const std::string& parent, const std::string& indent) const {
    std::string dot;
    for (const auto& cluster : _clusters) {
        if (cluster.parent != parent)
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
        dot += cluster_dot(cluster.id, inner);
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

    // Top-level nodes first, then the clusters (and the nodes they hold).
    for (const auto& node : _nodes)
        if (node.cluster.empty())
            dot += node_dot(node, indent);
    dot += cluster_dot("", indent);

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

    dot += "}\n";
    return dot;
}

} // namespace kotekan
