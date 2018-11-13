#include <random>

#include "fakeGpuPattern.hpp"
#include "visUtil.hpp"

// Register test patterns
REGISTER_FAKE_GPU_PATTERN(blockGpuPattern, "block");
REGISTER_FAKE_GPU_PATTERN(lostSamplesGpuPattern, "lostsamples");
REGISTER_FAKE_GPU_PATTERN(accumulateGpuPattern, "accumulate");
REGISTER_FAKE_GPU_PATTERN(gaussianGpuPattern, "gaussian");
REGISTER_FAKE_GPU_PATTERN(pulsarGpuPattern, "pulsar");


fakeGpuPattern::fakeGpuPattern(Config& config, const std::string& path)
{
    _num_elements = config.get<int>(path, "num_elements");
    _block_size = config.get<int>(path, "block_size");
    _samples_per_data_set = config.get<int>(path, "samples_per_data_set");
    _num_freq_in_frame = config.get_default<int>(path, "num_freq_in_frame", 1);
}


blockGpuPattern::blockGpuPattern(Config& config, const std::string& path) :
    fakeGpuPattern(config, path)
{

}


void blockGpuPattern::fill(gsl::span<int32_t>& data,
    chimeMetadata* metadata, int frame_number, int freq_id)
{

    int nb1 = _num_elements / _block_size;
    int num_blocks = nb1 * (nb1 + 1) / 2;

    DEBUG2("Block size %i, num blocks %i", _block_size, num_blocks);

    for (int b = 0; b < num_blocks; ++b){
        for (int y = 0; y < _block_size; ++y){
            for (int x = 0; x < _block_size; ++x) {
                int ind = b * _block_size * _block_size + x + y * _block_size;
                data[2 * ind + 0] = y;
                data[2 * ind + 1] = x;
            }
        }
    }
}


lostSamplesGpuPattern::lostSamplesGpuPattern(Config& config,
                                             const std::string& path) :
    fakeGpuPattern(config, path)
{

}

void lostSamplesGpuPattern::fill(gsl::span<int32_t>& data,
    chimeMetadata* metadata, int frame_number, int freq_id)
{

    uint32_t norm = _samples_per_data_set - frame_number;

    // Every frame has one more lost packet than the last
    for(int i = 0; i < _num_elements; i++) {
        for(int j = i; j < _num_elements; j++) {
            uint32_t bi = prod_index(i, j, _block_size, _num_elements);

            // The visibilities are row + 1j * col, scaled by the total number
            // of frames.
            data[2 * bi    ] = j * norm;  // Imag
            data[2 * bi + 1] = i * norm;  // Real
        }
    }

    metadata->lost_timesamples = frame_number;
}


accumulateGpuPattern::accumulateGpuPattern(Config& config,
                                           const std::string& path) :
    fakeGpuPattern(config, path)
{

}


void accumulateGpuPattern::fill(gsl::span<int32_t>& data,
    chimeMetadata* metadata, int frame_number, int freq_id)
{
    for(int i = 0; i < _num_elements; i++) {
        for(int j = i; j < _num_elements; j++) {
            uint32_t bi = prod_index(i, j, _block_size, _num_elements);

            // Every 4th sample the imaginary part is boosted by 4 * samples,
            // but we subtract off a constant to make it average the to be the
            // column index.
            data[2 * bi    ] =   // Imag
                (j + 4 * (frame_number % 4 == 0) - 1) * _samples_per_data_set;

            // ... similar for the real part, except we subtract every 4th
            // frame, and boost by a constant to ensure the average value is the
            // row.
            data[2 * bi + 1] =   // Real
                (i - 4 * ((frame_number + 1) % 4 == 0) + 1) * _samples_per_data_set;
        }
    }
}


gaussianGpuPattern::gaussianGpuPattern(Config& config,
                                       const std::string& path) :
    fakeGpuPattern(config, path)
{

}


void gaussianGpuPattern::fill(gsl::span<int32_t>& data,
    chimeMetadata* metadata, int frame_number, int freq_id)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> gaussian{0,1};

    float f_auto = pow(_samples_per_data_set, 0.5);
    float f_cross = pow(_samples_per_data_set / 2, 0.5);

    for(int i = 0; i < _num_elements; i++) {
        for(int j = i; j < _num_elements; j++) {
            uint32_t bi = prod_index(i, j, _block_size, _num_elements);

            if(i == j) {
                data[2 * bi + 1] =
                    _samples_per_data_set + (int32_t)(f_auto * gaussian(gen));
                data[2 * bi    ] = 0;
            } else {
                data[2 * bi + 1] = (int32_t)(f_cross * gaussian(gen));
                data[2 * bi    ] = (int32_t)(f_cross * gaussian(gen));
            }
        }
    }
}


pulsarGpuPattern::pulsarGpuPattern(Config& config,
                                   const std::string& path) :
    fakeGpuPattern(config, path)
{
    // set up pulsar polyco
    auto coeff = config.get<std::vector<float>>(path, "coeff");
    auto dm = config.get<float>(path, "dm");
    auto t_ref = config.get<double>(path, "t_ref");  // in days since MJD
    auto phase_ref = config.get<double>(path, "phase_ref");  // in number of rotations
    _rot_freq = config.get<double>(path, "rot_freq");  // in Hz
    _pulse_width = config.get<float>(path, "pulse_width");
    _polyco = Polyco(t_ref, dm, phase_ref, _rot_freq, coeff);
}


void pulsarGpuPattern::fill(gsl::span<int32_t>& data,
    chimeMetadata* metadata, int frame_number, int freq_id)
{
    // Fill frame with zeros
    std::fill(data.begin(), data.end(), 0);

    DEBUG2("GPS time %ds%dns", metadata->gps_time.tv_sec, metadata->gps_time.tv_nsec);

    // Figure out if we are in a pulse
    double toa = _polyco.next_toa(metadata->gps_time, freq_from_bin(freq_id));
    double last_toa = toa - 1. / _rot_freq;
    DEBUG2("TOA: %f, last TOA: %f", toa, last_toa);

    // TODO: CHIME specific
    // If so, add 10 to real part
    if (toa < _samples_per_data_set * 2.56e-6 || last_toa + _pulse_width > 0) {
        //DEBUG("Found pulse!");
        for(int i = 0; i < _num_elements; i++) {
            for(int j = i; j < _num_elements; j++) {
                uint32_t bi = prod_index(i, j, _block_size, _num_elements);
                data[2 * bi + 1] += 10 * _samples_per_data_set;
            }
        }
    }
}