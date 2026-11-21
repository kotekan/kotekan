#ifndef FENGINE_GAINS_HPP
#define FENGINE_GAINS_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Counter

#include <stdint.h> // for uint32_t
#include <string>   // for string

/**
 * @class fEngineGains
 * @brief Produces the per-bin gain tables cudaSparkFEngine applies before quantization.
 *
 * Three modes. "scalar" fills one table with the gain value; "file" reads one from
 * gain_file, fft_length/2 raw float32 values; both emit a single frame at startup and idle.
 * "auto" first emits a scalar table, then consumes the per-bin power frames the F-engine
 * stage accumulates (stats_buf, |X|^2 summed over stats_windows windows, pre-gain) and emits
 * a new table per stats frame with gain[k] = target_rms / rms(X[k]) per component, so every
 * bin's noise sits at target_rms quantizer counts (the CHORD bit-RMS rationale; 2 counts
 * leaves int4 headroom for RFI). A bin with no power gets gain_max. With gain_file_out the
 * newest auto table is also written as raw float32, loadable later through gain_file.
 *
 * @par Buffers
 * @buffer gain_buf   Gain tables
 *         @buffer_format fft_length / 2 float32
 *         @buffer_metadata none
 * @buffer stats_buf  Auto mode: per-bin power from cudaSparkFEngine
 *         @buffer_format fft_length / 2 float32
 *         @buffer_metadata none
 *
 * @conf fft_length    Int. The F-engine's window length; tables hold fft_length/2 bins.
 * @conf gain_mode     String, default "scalar". "scalar", "file" or "auto".
 * @conf gain          Float, default 1.0. The scalar table value, and auto's initial value.
 * @conf gain_file     String. File mode: raw float32 table to load.
 * @conf gain_file_out String, default none. Auto mode: write each table here.
 * @conf target_rms    Float, default 2.0. Auto mode: per-component bin RMS to aim for, in
 *                     quantizer counts.
 * @conf gain_max      Float, default 1e6. Auto mode: cap for empty bins.
 * @conf stats_windows Int, default 64. Windows per stats frame; must match the F-engine
 *                     stage.
 *
 * @par Metrics
 * @metric kotekan_fengine_gains_tables_total  Gain tables emitted.
 *
 * @author Andre Renard
 */
class fEngineGains : public kotekan::Stage {
public:
    fEngineGains(kotekan::Config& config, const std::string& unique_name,
                 kotekan::bufferContainer& buffer_container);
    ~fEngineGains() override = default;
    void main_thread() override;

private:
    /// Fills gain frame @p frame_id and marks it full. False on shutdown.
    bool emit_table(const float* table);

    Buffer* gain_buf;
    Buffer* stats_buf;

    uint64_t fft_length;
    std::string gain_mode;
    float gain;
    std::string gain_file;
    std::string gain_file_out;
    float target_rms;
    float gain_max;
    uint32_t stats_windows;

    int gain_frame_id = 0;

    kotekan::prometheus::Counter& tables_counter;
};

#endif /* FENGINE_GAINS_HPP */
