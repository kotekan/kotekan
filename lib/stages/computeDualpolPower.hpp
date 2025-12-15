#ifndef COMPUTE_DUALPOL_POWER
#define COMPUTE_DUALPOL_POWER

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer

#include <string>      // for string
#include <sys/types.h> // for uint

/**
 * @class computeDualpolPower
 * @brief Integrates VDIF voltage data into per-element power spectra.
 *
 * Consumes VDIF-formatted input frames (`vdif_in_buf`), squares and sums dual-pol complex samples
 * using AVX2, and writes integrated power and hit counts to `power_out_buf`. Integration is
 * performed over `power_integration_length` time samples, reducing
 * `samples_per_data_set -> samples_per_data_set / power_integration_length` time steps. Input data
 * are treated as packed 4+4-bit complex numbers immediately following a `VDIFHeader`; invalid
 * packets (header `invalid` bit) are skipped. The stage binds worker threads to the CPUs listed in
 * `cpu_affinity` and requires AVX2 for the fast kernel.
 *
 * @par Buffers
 * @buffer vdif_in_buf  Raw VDIF voltage frames.
 *     @buffer_format VDIF packets (header + 4-bit complex payload for each element/freq)
 *     @buffer_metadata none
 * @buffer power_out_buf Integrated power output.
 *     @buffer_format uint power values laid out [time][element][freq + count] (see source)
 *     @buffer_metadata none
 *
 * @conf samples_per_data_set     Int. Time samples per input frame.
 * @conf power_integration_length Int. Samples to integrate per output time.
 * @conf num_freq                 Int. Number of frequencies per element.
 * @conf num_elements             Int. Number of elements/polarizations.
 * @conf cpu_affinity             Array<Int>. CPU cores to pin worker threads to.
 *
 * @par Example
 * @code
 * compute_dualpol_power:
 *   kotekan_stage: computeDualpolPower
 *   vdif_in_buf: vdif_raw
 *   power_out_buf: vdif_power
 *   samples_per_data_set: 8000
 *   power_integration_length: 10
 *   num_freq: 1024
 *   num_elements: 2048
 *   cpu_affinity: [0, 1]
 * @endcode
 */

class computeDualpolPower : public kotekan::Stage {
public:
    computeDualpolPower(kotekan::Config& config, const std::string& unique_name,
                        kotekan::bufferContainer& buffer_container);
    ~computeDualpolPower();
    void main_thread() override;

private:
    inline void fastSqSumVdif(unsigned char* data, uint* temp_buf, uint* output);
    void parallelSqSumVdif(int loop_idx, int loop_length);
    Buffer* buf_in;
    Buffer* buf_out;

    int num_freq;
    int num_elem;
    int timesteps_in;
    int timesteps_out;
    int integration_length;
    unsigned int* integration_count;
    unsigned char* in_local;
    unsigned char* out_local;
};

#endif
