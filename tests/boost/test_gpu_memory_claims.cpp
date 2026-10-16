#define BOOST_TEST_MODULE "test_gpu_memory_claims"

#include "Config.hpp"             // for Config
#include "gpuDeviceInterface.hpp" // for gpuDeviceInterface, gpuMemoryOwnerScope
#include "gpuMemoryClaims.hpp"    // for classify_gpu_memory_claim, gpuClaimResult

#include <boost/test/included/unit_test.hpp>
#include <csignal>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>

// Two layers. Part 1 pins the policy, a pure function. Part 2 drives the real
// gpuDeviceInterface with a malloc-backed allocator -- it is compiled into this test because
// lib/gpu is built only when a GPU backend is found and the CI boost jobs are CPU-only, so
// without this the wiring (owner scopes, thread attribution, the array and view variants, the
// declared-share exemption) had no test that could fail.

namespace {

// FATAL_ERROR raises SIGTERM (exit_kotekan) before it throws. The throw is what these tests
// observe, so the signal is ignored for the life of the module.
struct ignore_sigterm {
    ignore_sigterm() {
        std::signal(SIGTERM, SIG_IGN);
    }
};
BOOST_GLOBAL_FIXTURE(ignore_sigterm);

class hostDevice final : public gpuDeviceInterface {
public:
    explicit hostDevice(kotekan::Config& config) : gpuDeviceInterface(config, "/test_device", 0) {}
    ~hostDevice() override {
        cleanup_memory();
    }

protected:
    void* alloc_gpu_memory(size_t len) override {
        return std::malloc(len);
    }
    void free_gpu_memory(void* p) override {
        std::free(p);
    }
};

const std::string A = "/gpu/stage_a";
const std::string B = "/gpu/stage_b";

/// The refusal this rule raises, as opposed to the allocator's own "already exists" or size
/// errors, which are also std::runtime_error.
bool refused_as_shared(const std::runtime_error& e) {
    return std::string(e.what()).find("is used by both") != std::string::npos;
}

} // namespace

// ---- part 1: the policy --------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(policy_first_claim_records_the_owner) {
    BOOST_CHECK(classify_gpu_memory_claim(nullptr, A, nullptr, nullptr)
                == gpuClaimResult::first_claim);
}

BOOST_AUTO_TEST_CASE(policy_the_owner_may_claim_again) {
    BOOST_CHECK(classify_gpu_memory_claim(&A, A, nullptr, nullptr) == gpuClaimResult::same_owner);
}

BOOST_AUTO_TEST_CASE(policy_a_second_stage_is_a_conflict) {
    BOOST_CHECK(classify_gpu_memory_claim(&A, B, nullptr, nullptr) == gpuClaimResult::conflict);
}

BOOST_AUTO_TEST_CASE(policy_a_share_needs_both_sides_and_one_handshake) {
    const std::string ring = "host_x_ringbuffer", other = "host_y_ringbuffer";
    BOOST_CHECK(classify_gpu_memory_claim(&A, B, &ring, &ring) == gpuClaimResult::shared);
    // one-sided: a declaration must not exempt a name for every stage on the device
    BOOST_CHECK(classify_gpu_memory_claim(&A, B, &ring, nullptr) == gpuClaimResult::conflict);
    BOOST_CHECK(classify_gpu_memory_claim(&A, B, nullptr, &ring) == gpuClaimResult::conflict);
    // two rings given one backing store: both declared, but not the same handshake
    BOOST_CHECK(classify_gpu_memory_claim(&A, B, &ring, &other) == gpuClaimResult::conflict);
}

// ---- part 2: the wiring --------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(wiring_a_second_stage_is_refused_at_construction) {
    kotekan::Config config;
    hostDevice dev(config);
    {
        gpuMemoryOwnerScope constructing(dev, A);
        BOOST_CHECK(dev.get_gpu_memory("scratch", 64) != nullptr);
    }
    gpuMemoryOwnerScope constructing(dev, B);
    BOOST_CHECK_EXCEPTION(dev.get_gpu_memory("scratch", 64), std::runtime_error, refused_as_shared);
}

BOOST_AUTO_TEST_CASE(wiring_the_owner_may_take_its_region_repeatedly) {
    kotekan::Config config;
    hostDevice dev(config);
    gpuMemoryOwnerScope constructing(dev, A);
    void* first = dev.get_gpu_memory("scratch", 64);
    void* again = nullptr;
    BOOST_CHECK_NO_THROW(again = dev.get_gpu_memory("scratch", 64));
    BOOST_CHECK(first == again);
}

BOOST_AUTO_TEST_CASE(wiring_arrays_are_checked_too) {
    // Per-frame intermediates are arrays; this is the chained-kernel case the rule exists for.
    kotekan::Config config;
    hostDevice dev(config);
    {
        gpuMemoryOwnerScope constructing(dev, A);
        BOOST_CHECK(dev.get_gpu_memory_array("frames", 0, 2, 64) != nullptr);
    }
    gpuMemoryOwnerScope constructing(dev, B);
    BOOST_CHECK_EXCEPTION(dev.get_gpu_memory_array("frames", 1, 2, 64), std::runtime_error,
                          refused_as_shared);
}

BOOST_AUTO_TEST_CASE(wiring_a_share_declared_by_both_through_one_ring_crosses_stages) {
    kotekan::Config config;
    hostDevice dev(config);
    void* producer = nullptr;
    void* consumer = nullptr;
    {
        gpuMemoryOwnerScope constructing(dev, A);
        dev.declare_shared_gpu_memory("ring", "host_ring");
        producer = dev.get_gpu_memory("ring", 64);
    }
    gpuMemoryOwnerScope constructing(dev, B);
    dev.declare_shared_gpu_memory("ring", "host_ring");
    BOOST_CHECK_NO_THROW(consumer = dev.get_gpu_memory("ring", 64));
    BOOST_CHECK(producer == consumer);
}

BOOST_AUTO_TEST_CASE(wiring_a_one_sided_declaration_does_not_exempt_the_name) {
    kotekan::Config config;
    hostDevice dev(config);
    {
        gpuMemoryOwnerScope constructing(dev, A);
        dev.declare_shared_gpu_memory("half", "shared_gpu_memory");
        dev.get_gpu_memory("half", 64);
    }
    gpuMemoryOwnerScope constructing(dev, B);
    BOOST_CHECK_EXCEPTION(dev.get_gpu_memory("half", 64), std::runtime_error, refused_as_shared);
}

BOOST_AUTO_TEST_CASE(wiring_two_rings_on_one_store_are_refused) {
    kotekan::Config config;
    hostDevice dev(config);
    {
        gpuMemoryOwnerScope constructing(dev, A);
        dev.declare_shared_gpu_memory("store", "host_ring_a");
        dev.get_gpu_memory("store", 64);
    }
    gpuMemoryOwnerScope constructing(dev, B);
    dev.declare_shared_gpu_memory("store", "host_ring_b");
    BOOST_CHECK_EXCEPTION(dev.get_gpu_memory("store", 64), std::runtime_error, refused_as_shared);
}

BOOST_AUTO_TEST_CASE(wiring_redeclaring_through_a_different_handshake_is_refused) {
    // A stage that declares one store through two rings has two rings on one store; and a
    // config declaration must not be silently replaced by a ring's.
    kotekan::Config config;
    hostDevice dev(config);
    gpuMemoryOwnerScope constructing(dev, A);
    dev.declare_shared_gpu_memory("store", "host_ring_a");
    BOOST_CHECK_NO_THROW(dev.declare_shared_gpu_memory("store", "host_ring_a"));
    BOOST_CHECK_THROW(dev.declare_shared_gpu_memory("store", "host_ring_b"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(wiring_a_declaration_outside_any_stage_is_refused) {
    kotekan::Config config;
    hostDevice dev(config);
    BOOST_CHECK_THROW(dev.declare_shared_gpu_memory("orphan", "host_ring"), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(wiring_a_registered_name_counts_before_allocation) {
    // A command that only takes its region in execute() registers the name from its
    // constructor, so the conflict still surfaces at construction.
    kotekan::Config config;
    hostDevice dev(config);
    {
        gpuMemoryOwnerScope constructing(dev, A);
        dev.register_gpu_memory_name("later");
    }
    gpuMemoryOwnerScope constructing(dev, B);
    BOOST_CHECK_EXCEPTION(dev.get_gpu_memory("later", 64), std::runtime_error, refused_as_shared);
}

BOOST_AUTO_TEST_CASE(wiring_two_stages_registering_one_name_conflict) {
    // The multi-pipe shape: neither stage has allocated yet, both registered from constructors.
    kotekan::Config config;
    hostDevice dev(config);
    {
        gpuMemoryOwnerScope constructing(dev, A);
        dev.register_gpu_memory_name("in");
    }
    gpuMemoryOwnerScope constructing(dev, B);
    BOOST_CHECK_EXCEPTION(dev.register_gpu_memory_name("in"), std::runtime_error,
                          refused_as_shared);
}

BOOST_AUTO_TEST_CASE(wiring_an_enqueuing_thread_is_its_stage) {
    // The backstop: a region first taken in execute() runs on the stage's own thread.
    kotekan::Config config;
    hostDevice dev(config);
    std::thread enqueuing([&] {
        dev.claim_memory_owner_thread(A);
        dev.get_gpu_memory("live", 64);
    });
    enqueuing.join();
    gpuMemoryOwnerScope constructing(dev, B);
    BOOST_CHECK_EXCEPTION(dev.get_gpu_memory("live", 64), std::runtime_error, refused_as_shared);
}

BOOST_AUTO_TEST_CASE(wiring_no_owner_means_no_check) {
    // Outside any scope, on an unregistered thread, there is no stage to attribute to -- and an
    // unowned take must not later count as somebody's claim either.
    kotekan::Config config;
    hostDevice dev(config);
    BOOST_CHECK_NO_THROW(dev.get_gpu_memory("unowned", 64));
    BOOST_CHECK_NO_THROW(dev.get_gpu_memory("unowned", 64));
    gpuMemoryOwnerScope constructing(dev, B);
    BOOST_CHECK_NO_THROW(dev.get_gpu_memory("unowned", 64));
}

BOOST_AUTO_TEST_CASE(wiring_a_view_belongs_to_the_stage_that_makes_it) {
    kotekan::Config config;
    hostDevice dev(config);
    {
        gpuMemoryOwnerScope constructing(dev, A);
        BOOST_CHECK(dev.get_gpu_memory_array("source", 0, 2, 64) != nullptr);
        dev.register_gpu_memory_name("reserved"); // A's name, but no region exists under it yet
    }
    gpuMemoryOwnerScope constructing(dev, B);
    // the source is A's
    BOOST_CHECK_EXCEPTION(dev.create_gpu_memory_array_view("source", 64, "view", 0, 32, 2),
                          std::runtime_error, refused_as_shared);
    // and so is a destination name A holds, whatever the source. The name is only registered,
    // not allocated, so the allocator's own "already exists" check cannot be what fires here.
    BOOST_CHECK(dev.get_gpu_memory_array("mine", 0, 2, 64) != nullptr);
    BOOST_CHECK_EXCEPTION(dev.create_gpu_memory_array_view("mine", 64, "reserved", 0, 32, 2),
                          std::runtime_error, refused_as_shared);
}

BOOST_AUTO_TEST_CASE(wiring_nested_construction_scopes_are_refused) {
    kotekan::Config config;
    hostDevice dev(config);
    gpuMemoryOwnerScope outer(dev, A);
    BOOST_CHECK_THROW(gpuMemoryOwnerScope inner(dev, B), std::runtime_error);
}
