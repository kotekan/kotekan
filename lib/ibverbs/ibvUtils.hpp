/**
 * @file
 * @brief Helpers shared by the ibverbs raw packet stages: device and protection domain setup,
 *        per-frame memory registration, raw packet QP state changes and completion channel waits.
 */
#ifndef IBV_UTILS_HPP
#define IBV_UTILS_HPP

#include "buffer.hpp" // for Buffer

#include <infiniband/verbs.h> // for ibv_context, ibv_pd, ibv_mr, ibv_qp, ibv_comp_channel
#include <stdint.h>           // for uint8_t, uint32_t
#include <string>             // for string
#include <vector>             // for vector

/**
 * @brief An open RDMA device and protection domain for one Ethernet netdev.
 *
 * Resolves the RDMA device behind a netdev through sysfs (e.g. enp1s0f0np0 -> rocep1s0f0), opens
 * it, allocates a PD and records the netdev's MAC address and MTU.
 *
 * @throw std::runtime_error when the netdev has no RDMA device, the port is not Ethernet, or any
 *        verbs call fails.
 */
class ibvDevice {
public:
    explicit ibvDevice(const std::string& netdev);
    ~ibvDevice();
    ibvDevice(const ibvDevice&) = delete;
    ibvDevice& operator=(const ibvDevice&) = delete;

    const std::string netdev;
    std::string rdma_name;
    ibv_context* ctx = nullptr;
    ibv_pd* pd = nullptr;
    const uint8_t port = 1;
    uint8_t mac[6];
    /// The netdev MTU: the largest packet after the Ethernet header.
    uint32_t mtu = 0;
    ibv_device_attr_ex attr;
    ibv_port_attr port_attr;
};

/**
 * @brief Memory regions covering every frame of a Buffer.
 *
 * Registers @c aligned_frame_size bytes of each frame once, in the constructor. Frames must stay
 * where they are: a stage that swaps frames between buffers would leave the registered regions
 * pointing at memory the buffer no longer owns, so @c lkey() checks the frame pointer every time.
 */
class ibvFrameMRs {
public:
    /// Registers every frame of @p buf with the ibverbs access flags @p access.
    /// @throw std::runtime_error when a registration fails.
    ibvFrameMRs(ibv_pd* pd, Buffer* buf, int access);
    ~ibvFrameMRs();
    ibvFrameMRs(const ibvFrameMRs&) = delete;
    ibvFrameMRs& operator=(const ibvFrameMRs&) = delete;

    /// The lkey of frame @p frame_id. FATAL_ERROR if the frame pointer has changed since
    /// registration.
    uint32_t lkey(int frame_id) const;

private:
    Buffer* buf;
    std::vector<ibv_mr*> mrs;
};

/**
 * @brief Move a raw packet QP from RESET through INIT and RTR to @p target (RTR or RTS).
 *
 * Raw packet QPs take only the state and, for INIT, the port number; any other attribute is
 * rejected.
 *
 * @throw std::runtime_error when a transition fails.
 */
void raw_qp_to_state(ibv_qp* qp, uint8_t port, ibv_qp_state target);

/// Create a completion channel whose fd is non-blocking, so a spurious wakeup cannot block
/// a stage in @c ibv_get_cq_event.
/// @throw std::runtime_error on failure.
ibv_comp_channel* create_nonblocking_comp_channel(ibv_context* ctx);

/**
 * @brief Wait up to @p timeout_ms for a completion event on @p channel and acknowledge it.
 *
 * The caller arms the CQ with @c ibv_req_notify_cq, polls it once more (a completion that landed
 * between the last poll and the arm would otherwise be missed), and only then waits here.
 *
 * @return true when an event arrived, false on timeout or a signal.
 */
bool wait_for_cq_event(ibv_comp_channel* channel, int timeout_ms);

#endif /* IBV_UTILS_HPP */
