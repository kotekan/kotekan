#define BOOST_TEST_MODULE "test_pipeline_graph"

#include "PipelineGraph.hpp" // for PipelineGraph, GraphNode, GraphCluster

#include <boost/test/included/unit_test.hpp>
#include <string>

using kotekan::PipelineGraph;

/// True if `needle` appears in `haystack`.
static bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

/// Every statement in a DOT body has to be terminated: catching a run-on line is
/// the whole point of rendering in one place, so check it structurally.
static void check_statements_terminated(const std::string& dot) {
    size_t start = 0;
    while (start < dot.size()) {
        size_t end = dot.find('\n', start);
        if (end == std::string::npos)
            end = dot.size();
        const std::string line = dot.substr(start, end - start);
        start = end + 1;
        // Skip blanks, comments, and the block delimiters.
        std::string trimmed = line;
        trimmed.erase(0, trimmed.find_first_not_of(" \t"));
        if (trimmed.empty() || trimmed.compare(0, 2, "//") == 0 || trimmed == "}"
            || trimmed == "digraph pipeline {" || contains(trimmed, "{"))
            continue;
        BOOST_CHECK_MESSAGE(trimmed.back() == ';', "unterminated statement: " + trimmed);
    }
}

BOOST_AUTO_TEST_CASE(nodes_and_edges) {
    PipelineGraph graph;
    graph.add_node("buffer")
        .add_line("buffer")
        .add_line("2/4 (50.0%)")
        .set_attr("shape", "ellipse");
    graph.add_node("/a/stage").add_line("/a/stage").set_attr("shape", "box");
    graph.add_edge("buffer", "/a/stage").set_attr("style", "dotted");

    const std::string dot = graph.to_dot();
    BOOST_CHECK(contains(dot, "digraph pipeline {"));
    // Label lines are joined, not concatenated.
    BOOST_CHECK(contains(dot, "label=<buffer<BR/>2/4 (50.0%)>"));
    BOOST_CHECK(contains(dot, "\"buffer\" -> \"/a/stage\" [style=\"dotted\"];"));
    check_statements_terminated(dot);
}

BOOST_AUTO_TEST_CASE(add_node_is_idempotent_and_references_stay_valid) {
    PipelineGraph graph;
    auto& first = graph.add_node("one");
    // Adding further nodes must not invalidate a reference already handed out --
    // stages hold on to theirs while adding the rest of their detail.
    for (int i = 0; i < 64; i++)
        graph.add_node("filler" + std::to_string(i));
    first.add_line("added later");
    // Same id gives back the same node rather than a duplicate.
    graph.add_node("one").set_attr("shape", "box");

    const std::string dot = graph.to_dot();
    BOOST_CHECK(contains(dot, "\"one\" [label=<added later>, shape=\"box\"];"));
}

BOOST_AUTO_TEST_CASE(labels_are_escaped) {
    PipelineGraph graph;
    // Buffer and stage names come from the config, so they can hold anything.
    graph.add_node("odd").add_line("a & b <c> \"d\"");
    graph.add_cluster("group").label = "x > y";
    graph.add_node("odd").cluster = "group";

    const std::string dot = graph.to_dot();
    BOOST_CHECK(contains(dot, "label=<a &amp; b &lt;c&gt; &quot;d&quot;>"));
    BOOST_CHECK(contains(dot, "label=<x &gt; y>;"));
    BOOST_CHECK(!contains(dot, "<c>"));
}

BOOST_AUTO_TEST_CASE(ids_are_quoted) {
    PipelineGraph graph;
    graph.add_node("say \"hi\"").add_line("quoted");
    const std::string dot = graph.to_dot();
    BOOST_CHECK(contains(dot, "\"say \\\"hi\\\"\""));
}

BOOST_AUTO_TEST_CASE(clusters_nest_and_hold_their_nodes) {
    PipelineGraph graph;
    auto& device = graph.add_cluster("/gpu/gpu_0");
    device.label = "GPU 0";
    device.set_attr("style", "filled");
    auto& mem = graph.add_cluster("/gpu/gpu_0/mem");
    mem.parent = "/gpu/gpu_0";

    graph.add_node("/gpu/gpu_0").add_line("gpu_0").cluster = "/gpu/gpu_0";
    graph.add_node("/gpu/gpu_0/mem/voltage").add_line("voltage").cluster = "/gpu/gpu_0/mem";
    graph.add_node("host_buffer").add_line("host_buffer");
    graph.add_edge("host_buffer", "/gpu/gpu_0");

    const std::string dot = graph.to_dot();
    const size_t device_open = dot.find("subgraph \"cluster_/gpu/gpu_0\" {");
    const size_t mem_open = dot.find("subgraph \"cluster_/gpu/gpu_0/mem\" {");
    const size_t anchor = dot.find("\"/gpu/gpu_0\" [");
    const size_t voltage = dot.find("\"/gpu/gpu_0/mem/voltage\" [");
    BOOST_REQUIRE(device_open != std::string::npos);
    BOOST_REQUIRE(mem_open != std::string::npos);
    // The memory region, the stage's anchor node and the GPU memory all sit
    // inside the device region; the anchor is what host buffer edges land on.
    BOOST_CHECK(device_open < mem_open);
    BOOST_CHECK(device_open < anchor && anchor < mem_open);
    BOOST_CHECK(mem_open < voltage);
    BOOST_CHECK(contains(dot, "\"host_buffer\" -> \"/gpu/gpu_0\";"));
    // A top level node is not swallowed by a cluster.
    BOOST_CHECK(dot.find("\"host_buffer\" [") < device_open);
    check_statements_terminated(dot);
}

BOOST_AUTO_TEST_CASE(dangling_edges_are_dropped_not_drawn) {
    PipelineGraph graph;
    graph.add_node("known").add_line("known");
    graph.add_edge("known", "never_added");
    graph.add_edge("never_added", "known");

    const std::string dot = graph.to_dot();
    // Graphviz would invent an empty node for the unknown endpoint, which reads
    // as a real (but nameless) part of the pipeline.
    BOOST_CHECK(!contains(dot, "\"known\" -> \"never_added\";"));
    BOOST_CHECK(!contains(dot, "\"never_added\" -> \"known\";"));
    BOOST_CHECK(contains(dot, "// dropped edge(s) referring to unknown node: never_added"));
}

BOOST_AUTO_TEST_CASE(graph_wide_attributes) {
    PipelineGraph graph;
    graph.header_comments.push_back("a pipeline");
    graph.graph_attrs["rankdir"] = "LR";
    graph.node_attrs["fontname"] = "Helvetica";
    graph.edge_attrs["fontsize"] = "8";
    graph.add_node("only").add_line("only");

    const std::string dot = graph.to_dot();
    BOOST_CHECK(contains(dot, "// a pipeline\n"));
    BOOST_CHECK(contains(dot, "graph [rankdir=\"LR\"];"));
    BOOST_CHECK(contains(dot, "node [fontname=\"Helvetica\"];"));
    BOOST_CHECK(contains(dot, "edge [fontsize=\"8\"];"));
    check_statements_terminated(dot);
}
