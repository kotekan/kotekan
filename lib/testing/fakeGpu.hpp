/*****************************************
@file
@brief Generate fake data in gpu buffer format.
- fakeGpu : public KotekanProcess
*****************************************/
#ifndef FAKE_GPU_HPP
#define FAKE_GPU_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "Telescope.hpp"       // for freq_id_t, Telescope, stream_t
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "datasetManager.hpp"  // for dset_id_t
#include "fakeGpuPattern.hpp"  // for FakeGpuPattern

#include <memory>   // for unique_ptr
#include <stdint.h> // for int32_t, uint32_t, uint64_t
#include <string>   // for string
#include <time.h>   // for timespec


/**
 * @class fakeGpu
 * @brief Generate fake data in gpu buffer format.
 *
 * Produce data formatted with the gpu buffer blocked packing.
 *
 * @par Buffers
 * @buffer out_buf The buffer of fake data.
 *         @buffer_format N2 GPU buffer format
 *         @buffer_metadata chordMetadata
 *
 * @conf  num_elements          Int. The number of elements (i.e. inputs) in
 *                              the correlator data
 * @conf  block_size            Int. The block size of the packed data
 * @conf  num_ev                Int. The number of eigenvectors to be stored
 * @conf  freq_id               Int. The single frequency ID to generate frames
 *                              for.
 * @conf  cadence               Float. The interval of time (in seconds)
 *                              between frames.
 * @conf  pre_accumulate        Bool. Simulate the GPU data before any
 *                              accumulation. This ignores any cadence setting.
 * @conf  samples_per_data_set  Int. FPGA seq ticks per frame. Only use for
 *                              `pre_accumulate`.
 * @conf  wait                  Bool. Sleep to try and output data at roughly
 *                              the correct cadence.
 * @conf  num_frames            Int. Exit after num_frames have been produced. If
 *                              less than zero, no limit is applied. Default
 *                              is `-1`.
 * @conf  pattern               String. Name of the pattern to fill with. These
 *                              patterns are registerd subclasses of
 *                              fakeGpuPattern.
 * @conf  drop_probability      Float. Probability that any individual frame gets
 *                              dropped. Default is zero, i.e. no frames are dropped.
 * @conf  dataset_id            Int. Use a fixed dataset ID. Otherwise, a meaningless
 *                              default value will be set.
 *
 * @note Look at the documentation for the test patterns to see any addtional
 *       configuration they require.
 *
 * @warning To work properly you must use the "fake" telescope type.
 *
 * @author Richard Shaw
 */
class FakeGpu : public kotekan::Stage {
public:
    FakeGpu(kotekan::Config& config, const std::string& unique_name,
            kotekan::bufferContainer& buffer_container);
    ~FakeGpu();
    void main_thread() override;


private:
    Buffer* out_buf;

    // Parameters read from the config
    int freq;
    float cadence;
    int32_t block_size;
    int32_t num_elements;
    int32_t samples_per_data_set;
    int32_t num_freq_in_frame;
    bool pre_accumulate;
    bool wait;
    int32_t num_frames;
    float drop_probability;
    dset_id_t dataset_id;

    // Pattern to use for filling
    std::unique_ptr<FakeGpuPattern> pattern;
};


/**
 * @brief A test telescope that just passes stream_id's straight through
 *
 * @conf  num_local_freq  The number of frequencies per stream.
 **/
class FakeTelescope : public Telescope {
public:
    FakeTelescope(const kotekan::Config& config, const std::string& path);

    // Dummy freq map implementations
    freq_id_t to_freq_id(stream_t stream_id, uint32_t ind) const override;
    double to_freq_MHz(freq_id_t freq_id) const override;
    double freq_width_MHz(freq_id_t freq_id) const override;
    size_t num_freq_per_stream() const override;
    size_t num_freq() const override;
    nyquist_zone_t nyquist_zone() const override;

    // Dummy time map implementations
    timespec to_time(uint64_t seq) const override;
    int64_t to_time_ns(uint64_t seq) const override;
    uint64_t to_seq(timespec time) const override;
    uint64_t seq_length_nsec() const override;
    bool gps_time_enabled() const override;
    station_id_t element_index_to_station_id(uint64_t el_idx, ElementOrder ord) const override;
    uint64_t station_id_to_element_index(station_id_t st_id, ElementOrder ord) const override;
    grid_idx_2d_t station_id_to_main_array_grid_indices(station_id_t st_id) const override; 
    vec3d_t station_id_to_feed_position_m(station_id_t st_id) const override;
    double get_feed_separation_x_m() const override;
    double get_feed_separation_y_m() const override;
    uint64_t get_grid_size_x() const override;
    uint64_t get_grid_size_y() const override;
    vec3d_t get_phase_center_in_grid_frame() const override;
private:
    size_t _num_local_freq;
};


#endif // FAKE_GPU
