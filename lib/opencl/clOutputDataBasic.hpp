#pragma once

#include "clCommand.hpp"
#include "buffer.hpp"
#include "chimeMetadata.hpp"
#include "errors.h"

#include <CL/cl.h>
#include <string>

class clOutputDataBasic : public clCommand {
public:
    clOutputDataBasic(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& host_buffers,
                 clDeviceInterface& device, int gpu_id);

    virtual cl_event execute(cl_event pre_event) override;

private:
    kotekan::Buffer* in_buf;   ///< The buffer this stage reads from
};
