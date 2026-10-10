/**
 * @file
 * @brief Optional NVTX ranges, for reading a kotekan run in Nsight Systems.
 *
 * Built with -DWITH_NVTX=ON (CUDA builds only) each KOTEKAN_NVTX_RANGE opens a range on the
 * calling thread that closes at the end of the enclosing scope. Without it the macro expands to
 * nothing and its argument is never evaluated (KOTEKAN_NVTX_PUSH/POP likewise), so a name that costs a format() is free in a
 * normal build. With no profiler attached NVTX3 calls return at once.
 */
#ifndef KOTEKAN_NVTX_HPP
#define KOTEKAN_NVTX_HPP

#ifdef KOTEKAN_NVTX
#include <nvtx3/nvToolsExt.h>

#include <string>

namespace kotekan {
class nvtxRange {
public:
    explicit nvtxRange(const char* name) {
        nvtxRangePushA(name);
    }
    explicit nvtxRange(const std::string& name) : nvtxRange(name.c_str()) {}
    ~nvtxRange() {
        nvtxRangePop();
    }
    nvtxRange(const nvtxRange&) = delete;
    nvtxRange& operator=(const nvtxRange&) = delete;
};
} // namespace kotekan

#define KOTEKAN_NVTX_CAT2(a, b) a##b
#define KOTEKAN_NVTX_CAT(a, b) KOTEKAN_NVTX_CAT2(a, b)
#define KOTEKAN_NVTX_RANGE(name)                                                                   \
    kotekan::nvtxRange KOTEKAN_NVTX_CAT(_kotekan_nvtx_range_, __LINE__)(name)
// For a step whose variables outlive any one block. Pair them on every path.
#define KOTEKAN_NVTX_PUSH(name) nvtxRangePushA(name)
#define KOTEKAN_NVTX_POP() nvtxRangePop()
#else
#define KOTEKAN_NVTX_RANGE(name)                                                                   \
    do {                                                                                           \
    } while (0)
#define KOTEKAN_NVTX_PUSH(name)                                                                    \
    do {                                                                                           \
    } while (0)
#define KOTEKAN_NVTX_POP()                                                                         \
    do {                                                                                           \
    } while (0)
#endif

#endif // KOTEKAN_NVTX_HPP
