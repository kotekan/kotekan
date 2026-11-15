#include "numaPolicy.hpp"

#include "kotekanLogging.hpp" // for WARN_NON_OO, ERROR_NON_OO

#include "fmt.hpp" // for format, fmt

#include <algorithm> // for find
#include <cerrno>    // for errno
#include <cstring>   // for strerror
#include <stddef.h>  // for size_t
#include <stdexcept> // for runtime_error

#ifdef WITH_NUMA
#include <numa.h>   // for numa_available, numa_node_of_cpu, numa_max_node, numa_allocate_nodemask
#include <numaif.h> // for get_mempolicy, set_mempolicy, MPOL_BIND, MPOL_DEFAULT
#endif

namespace kotekan {

#ifdef WITH_NUMA
namespace {

/// Renders a list of integers as "[a, b, c]" for log messages.
std::string list_to_string(const std::vector<int>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i)
        out += (i ? ", " : "") + std::to_string(values[i]);
    return out + "]";
}

} // namespace
#endif

bool numa_supported() {
#ifdef WITH_NUMA
    // numa_available() must be called before any other libnuma function, and
    // its answer does not change while the process runs.
    static const bool supported = numa_available() >= 0;
    return supported;
#else
    return false;
#endif
}

int numa_node_of_cpus(const std::vector<int>& cpus, const std::string& owner) {
#ifdef WITH_NUMA
    if (cpus.empty() || !numa_supported())
        return -1;

    // The distinct nodes, in the order the cores name them: the first is used.
    std::vector<int> nodes;
    for (int cpu : cpus) {
        const int node = numa_node_of_cpu(cpu);
        if (node < 0) {
            WARN_NON_OO("{:s}: core {:d} in cpu_affinity is not a CPU on this system ({:s}); "
                        "ignoring it when choosing a NUMA node",
                        owner, cpu, strerror(errno));
            continue;
        }
        if (std::find(nodes.begin(), nodes.end(), node) == nodes.end())
            nodes.push_back(node);
    }
    if (nodes.empty())
        return -1;
    if (nodes.size() > 1)
        WARN_NON_OO("{:s}: cpu_affinity {:s} spans NUMA nodes {:s}; placing its memory on node "
                    "{:d}, the node of the first core listed. Pin it to cores on one node, or "
                    "list a core on the intended node first, to choose.",
                    owner, list_to_string(cpus), list_to_string(nodes), nodes.front());
    return nodes.front();
#else
    (void)cpus;
    (void)owner;
    return -1;
#endif
}

ScopedNumaPolicy::ScopedNumaPolicy(int numa_node) {
#if defined(WITH_NUMA) && !defined(WITH_NO_MEMLOCK)
    if (numa_node < 0 || !numa_supported())
        return;
    if (numa_node > numa_max_node())
        throw std::runtime_error(
            fmt::format(fmt("Cannot place memory on NUMA node {:d}: the highest node on this "
                            "system is {:d}"),
                        numa_node, numa_max_node()));
    if (!save_current_policy())
        return;

    struct bitmask* node_mask = numa_allocate_nodemask();
    numa_bitmask_setbit(node_mask, numa_node);
    const int err = apply(MPOL_BIND, node_mask);
    numa_bitmask_free(node_mask);
    if (err != 0) {
        numa_bitmask_free(saved_mask);
        saved_mask = nullptr;
        throw std::runtime_error(
            fmt::format(fmt("Failed to bind memory allocation to NUMA node {:d}: {:s} ({:d})"),
                        numa_node, strerror(err), err));
    }
    restore_on_exit = true;
#else
    (void)numa_node;
#endif
}

ScopedNumaPolicy::ScopedNumaPolicy(system_default_t) {
#if defined(WITH_NUMA) && !defined(WITH_NO_MEMLOCK)
    if (!numa_supported() || !save_current_policy())
        return;
    if (saved_mode == MPOL_DEFAULT) {
        // Already there: nothing to change, and nothing to restore.
        numa_bitmask_free(saved_mask);
        saved_mask = nullptr;
        return;
    }
    const int err = apply(MPOL_DEFAULT, nullptr);
    if (err != 0) {
        numa_bitmask_free(saved_mask);
        saved_mask = nullptr;
        throw std::runtime_error(fmt::format(
            fmt("Failed to reset the NUMA memory policy to the system default: {:s} ({:d})"),
            strerror(err), err));
    }
    restore_on_exit = true;
#endif
}

ScopedNumaPolicy::~ScopedNumaPolicy() {
#if defined(WITH_NUMA) && !defined(WITH_NO_MEMLOCK)
    if (restore_on_exit) {
        // A destructor cannot throw; a policy left in place is not fatal, but
        // everything this thread allocates from here on goes to the wrong node.
        const int err = apply(saved_mode, saved_mask);
        if (err != 0)
            ERROR_NON_OO("Failed to restore the thread's NUMA memory policy: {:s} ({:d})",
                         strerror(err), err);
    }
    if (saved_mask != nullptr)
        numa_bitmask_free(saved_mask);
#endif
}

bool ScopedNumaPolicy::active() const {
    return restore_on_exit;
}

bool ScopedNumaPolicy::save_current_policy() {
#if defined(WITH_NUMA) && !defined(WITH_NO_MEMLOCK)
    // get_mempolicy() reports the mode with its flags, and the node mask, in
    // exactly the form set_mempolicy() takes them back.
    saved_mask = numa_allocate_nodemask();
    if (get_mempolicy(&saved_mode, saved_mask->maskp, saved_mask->size + 1, nullptr, 0) < 0) {
        WARN_NON_OO("Cannot read the thread's NUMA memory policy ({:s}); leaving it unchanged",
                    strerror(errno));
        numa_bitmask_free(saved_mask);
        saved_mask = nullptr;
        return false;
    }
    return true;
#else
    // Never read in this build; keeps clang's unused-private-field check quiet.
    (void)saved_mode;
    (void)saved_mask;
    return false;
#endif
}

int ScopedNumaPolicy::apply(int mode, const struct bitmask* mask) {
#if defined(WITH_NUMA) && !defined(WITH_NO_MEMLOCK)
    if (set_mempolicy(mode, mask ? mask->maskp : nullptr, mask ? mask->size + 1 : 0) < 0)
        return errno;
    return 0;
#else
    (void)mode;
    (void)mask;
    return 0;
#endif
}

} // namespace kotekan
