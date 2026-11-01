#include "ibvUtils.hpp"

#include "kotekanLogging.hpp" // for FATAL_ERROR_NON_OO, WARN_NON_OO
#include "udpPacket.hpp"      // for parse_mac

#include "fmt.hpp" // for format

#include <dirent.h>  // for opendir, readdir, closedir
#include <errno.h>   // for errno, EINTR, EAGAIN
#include <fcntl.h>   // for fcntl, O_NONBLOCK
#include <fstream>   // for ifstream
#include <poll.h>    // for poll, pollfd
#include <stdexcept> // for runtime_error
#include <string.h>  // for strerror, memset

namespace {

std::string read_sysfs(const std::string& path) {
    std::ifstream f(path);
    std::string value;
    if (!f || !std::getline(f, value))
        throw std::runtime_error(fmt::format(fmt("cannot read {:s}"), path));
    return value;
}

// The RDMA device behind a netdev, from /sys/class/net/<netdev>/device/infiniband/.
std::string rdma_device_of(const std::string& netdev) {
    const std::string dir = "/sys/class/net/" + netdev + "/device/infiniband";
    DIR* d = opendir(dir.c_str());
    if (!d)
        throw std::runtime_error(
            fmt::format(fmt("{:s} has no RDMA device ({:s}: {:s})"), netdev, dir, strerror(errno)));
    std::string name;
    while (struct dirent* e = readdir(d)) {
        if (e->d_name[0] != '.') {
            name = e->d_name;
            break;
        }
    }
    closedir(d);
    if (name.empty())
        throw std::runtime_error(
            fmt::format(fmt("{:s} has no RDMA device ({:s} is empty)"), netdev, dir));
    return name;
}

} // namespace

ibvDevice::ibvDevice(const std::string& _netdev) : netdev(_netdev) {
    rdma_name = rdma_device_of(netdev);
    if (!parse_mac(read_sysfs("/sys/class/net/" + netdev + "/address"), mac))
        throw std::runtime_error(fmt::format(fmt("cannot parse the MAC address of {:s}"), netdev));
    mtu = std::stoul(read_sysfs("/sys/class/net/" + netdev + "/mtu"));

    int num_devices = 0;
    ibv_device** devices = ibv_get_device_list(&num_devices);
    if (!devices)
        throw std::runtime_error(
            fmt::format(fmt("ibv_get_device_list failed: {:s}"), strerror(errno)));
    for (int i = 0; i < num_devices; i++) {
        if (rdma_name == ibv_get_device_name(devices[i])) {
            ctx = ibv_open_device(devices[i]);
            break;
        }
    }
    ibv_free_device_list(devices);
    if (!ctx)
        throw std::runtime_error(fmt::format(fmt("cannot open RDMA device {:s} for {:s}: {:s}"),
                                             rdma_name, netdev, strerror(errno)));

    std::string error;
    if (ibv_query_device_ex(ctx, nullptr, &attr))
        error = fmt::format(fmt("ibv_query_device_ex failed: {:s}"), strerror(errno));
    else if (ibv_query_port(ctx, port, &port_attr))
        error = fmt::format(fmt("ibv_query_port failed: {:s}"), strerror(errno));
    else if (port_attr.link_layer != IBV_LINK_LAYER_ETHERNET)
        error = fmt::format(fmt("{:s} port {:d} is not an Ethernet port"), rdma_name, port);
    else if (!(pd = ibv_alloc_pd(ctx)))
        error = fmt::format(fmt("ibv_alloc_pd failed: {:s}"), strerror(errno));
    if (!error.empty()) {
        ibv_close_device(ctx);
        throw std::runtime_error(error);
    }
    if (port_attr.state != IBV_PORT_ACTIVE)
        WARN_NON_OO("{:s} ({:s}) link is not up (port state {:d})", netdev, rdma_name,
                    (int)port_attr.state);
}

ibvDevice::~ibvDevice() {
    if (pd && ibv_dealloc_pd(pd))
        ERROR_NON_OO("ibv_dealloc_pd failed for {:s}: {:s}", rdma_name, strerror(errno));
    if (ctx && ibv_close_device(ctx))
        ERROR_NON_OO("ibv_close_device failed for {:s}: {:s}", rdma_name, strerror(errno));
}

ibvFrameMRs::ibvFrameMRs(ibv_pd* pd, Buffer* _buf, int access) : buf(_buf) {
    mrs.reserve(buf->num_frames);
    for (int i = 0; i < buf->num_frames; i++) {
        ibv_mr* mr = ibv_reg_mr(pd, buf->frames[i], buf->aligned_frame_size, access);
        if (!mr) {
            const int err = errno;
            for (ibv_mr* m : mrs)
                ibv_dereg_mr(m);
            throw std::runtime_error(fmt::format(
                fmt("ibv_reg_mr failed for frame {:d} of {:s} ({:d} bytes): {:s}. Check "
                    "ulimit -l and that the frames are not GPU memory."),
                i, buf->buffer_name, buf->aligned_frame_size, strerror(err)));
        }
        mrs.push_back(mr);
    }
}

ibvFrameMRs::~ibvFrameMRs() {
    for (ibv_mr* mr : mrs)
        if (ibv_dereg_mr(mr))
            ERROR_NON_OO("ibv_dereg_mr failed for {:s}: {:s}", buf->buffer_name, strerror(errno));
}

uint32_t ibvFrameMRs::lkey(int frame_id) const {
    if (buf->frames[frame_id] != mrs[frame_id]->addr)
        FATAL_ERROR_NON_OO("Frame {:d} of {:s} moved after it was registered with the NIC ({:p} "
                           "-> {:p}); a stage that swaps frames cannot share this buffer",
                           frame_id, buf->buffer_name, mrs[frame_id]->addr,
                           (void*)buf->frames[frame_id]);
    return mrs[frame_id]->lkey;
}

void raw_qp_to_state(ibv_qp* qp, uint8_t port, ibv_qp_state target) {
    ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = port;
    int rc = ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PORT);
    if (rc)
        throw std::runtime_error(
            fmt::format(fmt("ibv_modify_qp to INIT failed: {:s}"), strerror(rc)));
    attr.qp_state = IBV_QPS_RTR;
    rc = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
    if (rc)
        throw std::runtime_error(
            fmt::format(fmt("ibv_modify_qp to RTR failed: {:s}"), strerror(rc)));
    if (target == IBV_QPS_RTS) {
        attr.qp_state = IBV_QPS_RTS;
        rc = ibv_modify_qp(qp, &attr, IBV_QP_STATE);
        if (rc)
            throw std::runtime_error(
                fmt::format(fmt("ibv_modify_qp to RTS failed: {:s}"), strerror(rc)));
    }
}

ibv_comp_channel* create_nonblocking_comp_channel(ibv_context* ctx) {
    ibv_comp_channel* channel = ibv_create_comp_channel(ctx);
    if (!channel)
        throw std::runtime_error(
            fmt::format(fmt("ibv_create_comp_channel failed: {:s}"), strerror(errno)));
    int flags = fcntl(channel->fd, F_GETFL);
    if (flags < 0 || fcntl(channel->fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        const int err = errno;
        ibv_destroy_comp_channel(channel);
        throw std::runtime_error(fmt::format(
            fmt("cannot make the completion channel non-blocking: {:s}"), strerror(err)));
    }
    return channel;
}

bool wait_for_cq_event(ibv_comp_channel* channel, int timeout_ms) {
    pollfd pfd = {channel->fd, POLLIN, 0};
    int rc = poll(&pfd, 1, timeout_ms);
    if (rc < 0) {
        if (errno == EINTR)
            return false;
        FATAL_ERROR_NON_OO("poll on the completion channel failed: {:s}", strerror(errno));
    }
    if (rc == 0)
        return false;
    ibv_cq* cq;
    void* cq_context;
    if (ibv_get_cq_event(channel, &cq, &cq_context)) {
        if (errno == EAGAIN)
            return false;
        FATAL_ERROR_NON_OO("ibv_get_cq_event failed: {:s}", strerror(errno));
    }
    ibv_ack_cq_events(cq, 1);
    return true;
}
