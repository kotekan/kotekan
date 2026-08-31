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

BOOST_AUTO_TEST_CASE(clusters_holding_nothing_are_not_drawn) {
    PipelineGraph graph;
    // A section whose only stage drew itself somewhere else -- a gpuProcess
    // moving into its device's region -- would otherwise render as a labelled
    // empty rectangle, reading as a part of the pipeline that holds nothing.
    graph.add_cluster("/run_n2k").label = "run_n2k";
    auto& device = graph.add_cluster("__gpu/0");
    device.label = "GPU 0";
    graph.add_node("/run_n2k/gpu_0").add_line("gpu_0").cluster = device.id;

    // Empty all the way down: a box holding only empty boxes is just as empty.
    graph.add_cluster("/outer").label = "outer";
    graph.add_cluster("/outer/inner").parent = "/outer";

    const std::string dot = graph.to_dot();
    BOOST_CHECK(contains(dot, "subgraph \"cluster___gpu/0\" {"));
    BOOST_CHECK(!contains(dot, "cluster_/run_n2k"));
    BOOST_CHECK(!contains(dot, "cluster_/outer"));
    check_statements_terminated(dot);
}

BOOST_AUTO_TEST_CASE(a_cluster_counts_as_full_if_a_nested_one_holds_a_node) {
    PipelineGraph graph;
    graph.add_cluster("__gpu/0").label = "GPU 0";
    auto& work = graph.add_cluster("/run_n2k/gpu_0/work");
    work.parent = "__gpu/0";
    work.label = "run_n2k";
    auto& mem = graph.add_cluster("/run_n2k/gpu_0/mem");
    mem.parent = work.id;
    // The only node sits two levels down; nothing above it may be dropped.
    graph.add_node("/run_n2k/gpu_0/mem/voltage").add_line("voltage").cluster = mem.id;

    const std::string dot = graph.to_dot();
    for (const char* id :
         {"cluster___gpu/0", "cluster_/run_n2k/gpu_0/work", "cluster_/run_n2k/gpu_0/mem"})
        BOOST_CHECK_MESSAGE(contains(dot, id), std::string("dropped ") + id);
}

BOOST_AUTO_TEST_CASE(a_loop_in_the_cluster_parents_is_broken_not_followed) {
    PipelineGraph graph;
    // Cluster parents are set by whatever builds the graph, so a stage can wire
    // two as each other's parent. Rendering follows parent links, so a loop here
    // is an infinite recursion rather than a wrong-looking picture.
    auto& first = graph.add_cluster("first");
    auto& second = graph.add_cluster("second");
    first.parent = "second";
    second.parent = "first";
    graph.add_node("caught").add_line("caught").cluster = "first";
    // A cluster that merely hangs off the loop must not take the whole graph
    // down with it either.
    graph.add_cluster("below").parent = "second";
    graph.add_node("below_node").add_line("below_node").cluster = "below";
    graph.add_node("fine").add_line("fine");

    const std::string dot = graph.to_dot();
    // Nothing is lost: the loop costs the nesting, not the contents.
    BOOST_CHECK(contains(dot, "\"caught\" ["));
    BOOST_CHECK(contains(dot, "\"below_node\" ["));
    BOOST_CHECK(contains(dot, "\"fine\" ["));
    BOOST_CHECK(contains(dot, "// drawn at the top level, its parent chain loops: first"));
    BOOST_CHECK(contains(dot, "// drawn at the top level, its parent chain loops: second"));
    check_statements_terminated(dot);
}

BOOST_AUTO_TEST_CASE(common_cluster_survives_a_parent_loop) {
    PipelineGraph graph;
    graph.add_cluster("first").parent = "second";
    graph.add_cluster("second").parent = "first";
    graph.add_node("a").cluster = "first";
    graph.add_node("b").cluster = "second";
    // A cluster hanging below the loop reaches it through its own ancestry.
    graph.add_cluster("below").parent = "second";
    graph.add_node("c").cluster = "below";
    graph.add_node("outside");

    // Before the guard this walked the loop forever. A node inside (or below)
    // a loop has no meaningful enclosing cluster, so it reads as top level.
    BOOST_CHECK_EQUAL(graph.common_cluster({"a"}), "");
    BOOST_CHECK_EQUAL(graph.common_cluster({"a", "b"}), "");
    BOOST_CHECK_EQUAL(graph.common_cluster({"c"}), "");
    BOOST_CHECK_EQUAL(graph.common_cluster({"a", "outside"}), "");
}

BOOST_AUTO_TEST_CASE(json_breaks_cluster_parent_loops_too) {
    PipelineGraph graph;
    graph.add_cluster("first").parent = "second";
    graph.add_cluster("second").parent = "first";
    graph.add_node("a").add_line("a").cluster = "first";
    graph.add_cluster("sane").label = "sane";
    graph.add_node("b").add_line("b").cluster = "sane";

    // Same rule as the DOT output: a consumer recursing on parents the way
    // cluster_dot() does must not inherit the loop.
    const nlohmann::json out = graph.to_json();
    BOOST_REQUIRE(out.contains("cyclic_clusters"));
    BOOST_CHECK_EQUAL(out["cyclic_clusters"].size(), 2u);
    for (const auto& cluster : out["clusters"]) {
        if (cluster["id"] == "first" || cluster["id"] == "second")
            BOOST_CHECK_EQUAL(cluster["parent"], "");
    }
    // A graph with sound wiring does not carry the key at all.
    PipelineGraph sound;
    sound.add_node("only").add_line("only");
    BOOST_CHECK(!sound.to_json().contains("cyclic_clusters"));
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

BOOST_AUTO_TEST_CASE(common_cluster_places_a_node_with_what_touches_it) {
    PipelineGraph graph;
    // section
    //   +-- device_0 (stage_a)
    //   +-- device_1 (stage_b)
    // stage_c is outside the section entirely.
    graph.add_cluster("section");
    graph.add_cluster("device_0").parent = "section";
    graph.add_cluster("device_1").parent = "section";
    graph.add_node("stage_a").cluster = "device_0";
    graph.add_node("stage_a2").cluster = "device_0";
    graph.add_node("stage_b").cluster = "device_1";
    graph.add_node("stage_c");

    // Everything on one device: the buffer is private to it.
    BOOST_CHECK_EQUAL(graph.common_cluster({"stage_a", "stage_a2"}), "device_0");
    // Shared across the section's devices: it belongs to the section.
    BOOST_CHECK_EQUAL(graph.common_cluster({"stage_a", "stage_b"}), "section");
    // Shared with a stage outside: it stays at the top level.
    BOOST_CHECK_EQUAL(graph.common_cluster({"stage_a", "stage_c"}), "");
    // An unknown node is at the top level as far as anyone can tell.
    BOOST_CHECK_EQUAL(graph.common_cluster({"stage_a", "no_such_node"}), "");
    BOOST_CHECK_EQUAL(graph.common_cluster({}), "");
}

BOOST_AUTO_TEST_CASE(stage_categories) {
    using kotekan::GraphCategory;
    // The GPU meta-stages are named exactly.
    BOOST_CHECK(kotekan::stage_category("cudaProcess") == GraphCategory::Gpu);
    BOOST_CHECK(kotekan::stage_category("hipProcess") == GraphCategory::Gpu);
    // I/O is matched on substrings, so unseen stages still land right.
    BOOST_CHECK(kotekan::stage_category("bufferSend") == GraphCategory::Io);
    BOOST_CHECK(kotekan::stage_category("frbNetworkProcess") == GraphCategory::Io);
    BOOST_CHECK(kotekan::stage_category("hdf5FileRead") == GraphCategory::Io);
    // Anything else is compute.
    BOOST_CHECK(kotekan::stage_category("accumulate") == GraphCategory::Compute);
    BOOST_CHECK(kotekan::stage_category("") == GraphCategory::Compute);
}

BOOST_AUTO_TEST_CASE(leaf_names_and_byte_sizes) {
    BOOST_CHECK_EQUAL(kotekan::leaf_name("/gen/voltage"), "voltage");
    BOOST_CHECK_EQUAL(kotekan::leaf_name("/copy"), "copy");
    BOOST_CHECK_EQUAL(kotekan::leaf_name("copy"), "copy");
    // A trailing slash leaves nothing to shorten to.
    BOOST_CHECK_EQUAL(kotekan::leaf_name("/gen/"), "/gen/");

    BOOST_CHECK_EQUAL(kotekan::human_bytes(0), "0 B");
    BOOST_CHECK_EQUAL(kotekan::human_bytes(1023), "1023 B");
    BOOST_CHECK_EQUAL(kotekan::human_bytes(1024), "1 KiB");
    BOOST_CHECK_EQUAL(kotekan::human_bytes(32 * 1024), "32 KiB");
    BOOST_CHECK_EQUAL(kotekan::human_bytes(size_t(3) * 1024 * 1024 * 1024), "3 GiB");
}

BOOST_AUTO_TEST_CASE(json_carries_the_same_graph) {
    PipelineGraph graph;
    graph.graph_attrs["rankdir"] = "LR";
    graph.add_cluster("section").label = "gen";
    graph.add_node("stage").add_line("voltage").add_line("<testDataGen>").cluster = "section";
    graph.add_node("buffer").add_line("host_voltage_buffer").set_attr("shape", "box");
    graph.add_edge("stage", "buffer").set_attr("weight", "4");
    graph.add_edge("stage", "never_added");

    const nlohmann::json out = graph.to_json();
    BOOST_CHECK_EQUAL(out["graph_attrs"]["rankdir"], "LR");
    BOOST_REQUIRE_EQUAL(out["clusters"].size(), 1u);
    BOOST_CHECK_EQUAL(out["clusters"][0]["label"], "gen");
    BOOST_REQUIRE_EQUAL(out["nodes"].size(), 2u);
    BOOST_CHECK_EQUAL(out["nodes"][0]["id"], "stage");
    BOOST_CHECK_EQUAL(out["nodes"][0]["cluster"], "section");
    // The label stays in pieces, for a client that lays the graph out itself.
    BOOST_REQUIRE_EQUAL(out["nodes"][0]["label_lines"].size(), 2u);
    BOOST_CHECK_EQUAL(out["nodes"][0]["label_lines"][1], "<testDataGen>");
    // Same rule as the DOT output: an edge to a node that was never added is
    // not a fact about the pipeline.
    BOOST_REQUIRE_EQUAL(out["edges"].size(), 1u);
    BOOST_CHECK_EQUAL(out["edges"][0]["to"], "buffer");
    BOOST_CHECK_EQUAL(out["edges"][0]["attrs"]["weight"], "4");
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

BOOST_AUTO_TEST_CASE(options_are_read_from_query_arguments) {
    using kotekan::GraphOptions;

    // Nothing asked for leaves every default in place.
    const GraphOptions defaults = GraphOptions::from_query({});
    BOOST_CHECK(defaults.cluster && defaults.legend && defaults.pools && defaults.runtime);
    BOOST_CHECK_EQUAL(defaults.rankdir, "LR");

    const GraphOptions chosen = GraphOptions::from_query({{"legend", "0"},
                                                          {"runtime", "false"},
                                                          {"urls", "off"},
                                                          {"kernels", "on"},
                                                          {"cluster", ""},
                                                          {"rankdir", "TB"}});
    BOOST_CHECK(!chosen.legend);
    BOOST_CHECK(!chosen.runtime);
    BOOST_CHECK(!chosen.urls); // any value that is not an "on" form turns it off
    BOOST_CHECK(chosen.kernels);
    BOOST_CHECK(chosen.cluster); // a bare `?cluster` counts as asking for it
    BOOST_CHECK(chosen.pools);   // not mentioned, stays at its default
    BOOST_CHECK_EQUAL(chosen.rankdir, "TB");

    // A direction dot does not know cannot reach the layout.
    BOOST_CHECK_EQUAL(GraphOptions::from_query({{"rankdir", "sideways"}}).rankdir, "LR");
}

BOOST_AUTO_TEST_CASE(default_style_and_legend_render) {
    PipelineGraph graph;
    graph.options.rankdir = "BT";
    graph.apply_default_style();
    graph.add_legend();

    const std::string dot = graph.to_dot();
    // The style follows the direction the options asked for.
    BOOST_CHECK(contains(dot, "rankdir=\"BT\""));
    // Everything graphviz would leave black carries an explicit colour.
    BOOST_CHECK(contains(dot, "fontcolor"));
    // The key is a cluster of one node per category, each in its colours.
    BOOST_CHECK(contains(dot, "subgraph \"cluster___legend\""));
    BOOST_CHECK(contains(dot, "stage on the CPU"));
    BOOST_CHECK(contains(dot, "the far end: a socket, a port, a file"));
    check_statements_terminated(dot);
}
