#ifndef GPU_MEMORY_CLAIMS_HPP
#define GPU_MEMORY_CLAIMS_HPP

#include <string>

/// The outcome of one stage's claim on a named GPU memory region.
enum class gpuClaimResult {
    first_claim, ///< no owner yet: record this stage
    same_owner,  ///< the owning stage again -- another command of it, or a later frame
    shared,      ///< both stages declared the share through the same handshake
    conflict,    ///< anything else: nothing known orders the two stages' accesses
};

/**
 * @brief Decide whether a stage may use a named GPU memory region another stage already owns.
 *
 * A named region belongs to one stage. Two stages may share it only if BOTH declare the share,
 * and name the SAME thing as what orders their accesses -- the handshake. For a GPU ring buffer
 * that is the host RingBuffer both sides wait on and signal, and the ring classes declare it
 * themselves; for anything else a stage lists the name under `shared_gpu_memory`, and the
 * handshake is that key. Requiring both sides is what keeps a declaration from exempting a
 * name for every stage on the device; requiring the same handshake is what refuses two rings
 * that were given one backing store by mistake.
 *
 * Nothing finer is decidable here. Whether two stages' accesses are ordered depends on which
 * streams each kernel and copy lands on and on the chains being enqueued contiguously, none of
 * which is visible where a region is handed out; and "same cuda_stream_base" is not a proxy
 * (a chain that crosses streams can be overtaken, and an explicit cuda_stream ignores the base).
 *
 * @param existing_owner     the recorded owner, or nullptr for a first claim.
 * @param owner              the claiming stage.
 * @param existing_handshake the owner's declaration for this name, or nullptr if none.
 * @param claimant_handshake the claimant's declaration for this name, or nullptr if none.
 */
inline gpuClaimResult classify_gpu_memory_claim(const std::string* existing_owner,
                                                const std::string& owner,
                                                const std::string* existing_handshake,
                                                const std::string* claimant_handshake) {
    if (existing_owner == nullptr)
        return gpuClaimResult::first_claim;
    if (*existing_owner == owner)
        return gpuClaimResult::same_owner;
    if (existing_handshake && claimant_handshake && *existing_handshake == *claimant_handshake)
        return gpuClaimResult::shared;
    return gpuClaimResult::conflict;
}

#endif // GPU_MEMORY_CLAIMS_HPP
