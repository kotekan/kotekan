#ifndef BUFFER_SPLIT_HPP
#define BUFFER_SPLIT_HPP

#include <vector>
#include <string>

#include "buffer.h"
#include "KotekanProcess.hpp"

class BufferSplit : public KotekanProcess {
public:

    /// Constructor
    BufferSplit(Config& config,
                const string& unique_name,
                bufferContainer &buffer_container);

    /// Destructor
    ~BufferSplit();

    /// Deprecated
    void apply_config(uint64_t fpga_seq) override;

    /// Thread for merging the frames.
    void main_thread() override;
private:

    /// Array of input buffers to get frames from
    struct Buffer* in_buf;

    /// The output buffer to put frames into
    std::vector<struct Buffer*> out_bufs;
};

#endif