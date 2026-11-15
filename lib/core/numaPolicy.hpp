/**
 * @file
 * @brief NUMA placement of the objects kotekan builds: which node a config
 *        block's memory belongs on, and a scoped memory policy to put it there.
 *  - numa_supported
 *  - numa_node_of_cpus
 *  - ScopedNumaPolicy
 */

#ifndef NUMA_POLICY_HPP
#define NUMA_POLICY_HPP

#include <string> // for string
#include <vector> // for vector

struct bitmask; // libnuma's node mask, kept out of this header

namespace kotekan {

/**
 * @brief Whether memory can be placed on NUMA nodes at all.
 *
 * @return true when kotekan was built with libnuma (``USE_NUMA``) and the
 *         running kernel exposes a NUMA topology. When false, every placement
 *         request below is a no-op.
 */
bool numa_supported();

/**
 * @brief The NUMA node the given CPU cores are on.
 *
 * Used to place a stage's memory next to the cores its @c cpu_affinity pins
 * it to. When the cores span more than one node a warning naming @p owner is
 * logged, and the node of the first core listed is used.
 *
 * @param cpus  CPU core numbers, zero based (a @c cpu_affinity list).
 * @param owner What the cores belong to, for the log messages (a stage's
 *              unique name).
 * @return The node of the first core listed, or -1 when the list is empty,
 *         NUMA is unavailable, or none of the cores maps to a node.
 */
int numa_node_of_cpus(const std::vector<int>& cpus, const std::string& owner);

/**
 * @brief Binds the calling thread's memory allocations to one NUMA node for as
 *        long as the object is alive, then restores the previous policy.
 *
 * Kotekan is run with the kernel's automatic NUMA balancing disabled, so a
 * page stays on the node it was first allocated on. The thread building the
 * pipeline is not pinned, so without this an object lands on whichever node
 * that thread happened to be running on. Wrapping a constructor in a
 * ScopedNumaPolicy puts the object, and everything the constructor allocates,
 * on the node its threads will run on.
 *
 * The policy is @c MPOL_BIND: allocations come from the given node only, as
 * they do for buffer frames. It covers memory this thread first touches while
 * the object is alive, and is inherited by any thread created in that time.
 * Scopes nest: the destructor restores whatever policy was in effect when the
 * object was created, not the system default.
 *
 * A no-op when built without libnuma (``USE_NUMA=OFF``) or with
 * ``NO_MEMLOCK=ON``, when the kernel has no NUMA support, or when the node is
 * negative.
 */
class ScopedNumaPolicy {
public:
    /**
     * @brief Bind this thread's allocations to @p numa_node.
     *
     * @param numa_node The node to allocate on; negative leaves the policy
     *                  untouched.
     * @throws std::runtime_error if the node does not exist on this system, or
     *         the kernel refuses the policy (e.g. the node has no memory).
     */
    explicit ScopedNumaPolicy(int numa_node);

    /// Tag selecting the system default policy, see the constructor below.
    struct system_default_t {};
    static constexpr system_default_t system_default{};

    /**
     * @brief Return this thread to the system default policy (allocate on the
     *        node of the CPU doing the first touch) for the scope.
     *
     * For handing control to code that places its own memory and starts its
     * own threads, such as the DPDK EAL, from inside a bound scope.
     */
    explicit ScopedNumaPolicy(system_default_t);

    ~ScopedNumaPolicy();

    ScopedNumaPolicy(const ScopedNumaPolicy&) = delete;
    ScopedNumaPolicy& operator=(const ScopedNumaPolicy&) = delete;

    /// @return true when this object changed the thread's policy, and so will
    ///         restore it.
    bool active() const;

private:
    /// Saves the thread's current policy for the destructor to restore.
    /// @return false, after logging a warning, if it could not be read.
    bool save_current_policy();

    /// Sets the thread's policy. @return 0 on success, else the errno.
    static int apply(int mode, const struct bitmask* mask);

    bool restore_on_exit = false;
    int saved_mode = 0;
    struct bitmask* saved_mask = nullptr;
};

} // namespace kotekan

#endif /* NUMA_POLICY_HPP */
