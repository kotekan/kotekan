#ifndef HSA_EVENT_CONTAINER_H
#define HSA_EVENT_CONTAINER_H

#include "gpuEventContainer.hpp"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_finalize.h"

class hsaEventContainer final : public gpuEventContainer {

public:
    void set(void* sig) override;
    void* get() override;
    void unset() override;
    void wait() override;

private:
    hsa_signal_t signal;
};

#endif // HSA_EVENT_CONTAINER