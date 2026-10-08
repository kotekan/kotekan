#define BOOST_TEST_MODULE "test_numa_policy"

#include <boost/test/included/unit_test.hpp>
#include <stdexcept> // for runtime_error
#include <string>    // for string
#include <vector>    // for vector

#ifdef __linux__
#include <algorithm>         // for find
#include <cstring>           // for memset
#include <linux/mempolicy.h> // for MPOL_BIND, MPOL_DEFAULT, MPOL_F_NODE, MPOL_F_ADDR
#include <sys/mman.h>        // for mmap, munmap
#include <sys/syscall.h>     // for SYS_get_mempolicy
#include <unistd.h>          // for syscall
#endif

// the code to test:
#include "numaPolicy.hpp" // for ScopedNumaPolicy, numa_node_of_cpus, numa_supported

using kotekan::numa_node_of_cpus;
using kotekan::numa_supported;
using kotekan::ScopedNumaPolicy;

// Only the builds that can set a policy have a use for reading one back; a
// NO_MEMLOCK build never sets any, and its unused probe would fail -Werror.
#if defined(__linux__) && !defined(WITH_NO_MEMLOCK)
namespace {

/// The calling thread's memory policy, read straight from the kernel so that
/// the test does not depend on libnuma: the mode, and the first 64 nodes of
/// its node mask.
struct ThreadPolicy {
    int mode = -1;
    unsigned long nodes = 0;
};

ThreadPolicy thread_policy() {
    ThreadPolicy policy;
    unsigned long mask[16] = {0}; // room for 1024 nodes, more than any kernel has
    const long ret = syscall(SYS_get_mempolicy, &policy.mode, mask, sizeof(mask) * 8, nullptr, 0);
    BOOST_REQUIRE_EQUAL(ret, 0);
    policy.nodes = mask[0];
    return policy;
}

} // namespace
#endif

BOOST_AUTO_TEST_CASE(node_of_cpus) {
    BOOST_CHECK_EQUAL(numa_node_of_cpus({}, "test"), -1);

    if (!numa_supported()) {
        BOOST_TEST_MESSAGE("NUMA not available: cores cannot be mapped to nodes");
        BOOST_CHECK_EQUAL(numa_node_of_cpus({0}, "test"), -1);
        return;
    }

    const int node_of_core_0 = numa_node_of_cpus({0}, "test");
    BOOST_CHECK_GE(node_of_core_0, 0);
    BOOST_CHECK_EQUAL(numa_node_of_cpus({0, 0}, "test"), node_of_core_0);

    // A core that does not exist is ignored (with a warning), not mapped.
    const int bogus_core = 1 << 20;
    BOOST_CHECK_EQUAL(numa_node_of_cpus({bogus_core}, "test"), -1);
    BOOST_CHECK_EQUAL(numa_node_of_cpus({bogus_core, 0}, "test"), node_of_core_0);

    // On a machine with more than one node, a list spanning two nodes resolves
    // to the node of the first core listed, whichever order they come in.
    int other_core = -1;
    for (int core = 1; core < 4096 && other_core < 0; ++core) {
        const int node = numa_node_of_cpus({core}, "test");
        if (node < 0)
            break; // ran off the end of the CPUs
        if (node != node_of_core_0)
            other_core = core;
    }
    if (other_core < 0) {
        BOOST_TEST_MESSAGE("single NUMA node: nothing to span");
        return;
    }
    const int other_node = numa_node_of_cpus({other_core}, "test");
    BOOST_CHECK_EQUAL(numa_node_of_cpus({0, other_core}, "test"), node_of_core_0);
    BOOST_CHECK_EQUAL(numa_node_of_cpus({other_core, 0}, "test"), other_node);
}

#ifdef __linux__
// The scoped policy binds this thread's allocations to the node, nests, and
// restores what was there before.
BOOST_AUTO_TEST_CASE(scoped_policy_binds_and_restores) {
#ifdef WITH_NO_MEMLOCK
    BOOST_TEST_MESSAGE("NO_MEMLOCK build: memory policies are never set");
    ScopedNumaPolicy bind(0);
    BOOST_CHECK(!bind.active());
#else
    if (!numa_supported()) {
        BOOST_TEST_MESSAGE("NUMA not available: memory policies are never set");
        ScopedNumaPolicy bind(0);
        BOOST_CHECK(!bind.active());
        return;
    }

    const ThreadPolicy before = thread_policy();
    const int node = numa_node_of_cpus({0}, "test");
    BOOST_REQUIRE_GE(node, 0);
    BOOST_REQUIRE_LT(node, 64); // the one mask word the test looks at
    {
        ScopedNumaPolicy bind(node);
        BOOST_CHECK(bind.active());
        ThreadPolicy inside = thread_policy();
        BOOST_CHECK_EQUAL(inside.mode, (int)MPOL_BIND);
        BOOST_CHECK_EQUAL(inside.nodes, 1UL << node);

        {
            // Handing off to code that places its own memory: back to the
            // default for the inner scope only.
            ScopedNumaPolicy own_placement(ScopedNumaPolicy::system_default);
            BOOST_CHECK(own_placement.active());
            BOOST_CHECK_EQUAL(thread_policy().mode, (int)MPOL_DEFAULT);
        }
        inside = thread_policy();
        BOOST_CHECK_EQUAL(inside.mode, (int)MPOL_BIND);
        BOOST_CHECK_EQUAL(inside.nodes, 1UL << node);

        // A negative node leaves the binding in place.
        ScopedNumaPolicy none(-1);
        BOOST_CHECK(!none.active());
        BOOST_CHECK_EQUAL(thread_policy().mode, (int)MPOL_BIND);
    }
    ThreadPolicy after = thread_policy();
    BOOST_CHECK_EQUAL(after.mode, before.mode);
    BOOST_CHECK_EQUAL(after.nodes, before.nodes);

    // A node this machine does not have is a config error, and leaves the
    // policy alone.
    BOOST_CHECK_THROW((void)ScopedNumaPolicy(1 << 20), std::runtime_error);
    after = thread_policy();
    BOOST_CHECK_EQUAL(after.mode, before.mode);
    BOOST_CHECK_EQUAL(after.nodes, before.nodes);

    // Already on the default policy: nothing to change, nothing to restore.
    if (before.mode == (int)MPOL_DEFAULT) {
        ScopedNumaPolicy own_placement(ScopedNumaPolicy::system_default);
        BOOST_CHECK(!own_placement.active());
    }
#endif
}

// Pages first touched inside a scope land on the bound node: on every node
// with CPUs, so not merely on the node this thread happens to run on.
BOOST_AUTO_TEST_CASE(scoped_policy_places_allocations) {
#ifndef WITH_NO_MEMLOCK
    if (!numa_supported()) {
        BOOST_TEST_MESSAGE("NUMA not available: nothing to place");
        return;
    }
    std::vector<int> nodes;
    for (int core = 0; core < 4096; ++core) {
        const int node = numa_node_of_cpus({core}, "test");
        if (node < 0)
            break; // ran off the end of the CPUs
        if (std::find(nodes.begin(), nodes.end(), node) == nodes.end())
            nodes.push_back(node);
    }
    BOOST_REQUIRE(!nodes.empty());

    const size_t size = 8 << 20;
    for (int node : nodes) {
        ScopedNumaPolicy bind(node);
        BOOST_REQUIRE(bind.active());
        // Fresh anonymous pages, so every one of them is first touched here.
        void* block =
            mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        BOOST_REQUIRE(block != MAP_FAILED);
        memset(block, 1, size);
        for (size_t offset = 0; offset < size; offset += 1 << 20) {
            int page_node = -1;
            const long ret = syscall(SYS_get_mempolicy, &page_node, nullptr, 0,
                                     (char*)block + offset, MPOL_F_NODE | MPOL_F_ADDR);
            BOOST_REQUIRE_EQUAL(ret, 0);
            BOOST_CHECK_EQUAL(page_node, node);
        }
        munmap(block, size);
    }
#else
    BOOST_TEST_MESSAGE("NO_MEMLOCK build: memory policies are never set");
#endif
}
#endif
