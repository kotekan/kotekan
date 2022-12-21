#include "gpuSimulateCudaBasebandBeamformer.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, mark_frame_empty, mark_frame_full, pass_metadata
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO, DEBUG

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_KOTEKAN_STAGE(gpuSimulateCudaBasebandBeamformer);


gpuSimulateCudaBasebandBeamformer::gpuSimulateCudaBasebandBeamformer(Config& config, const std::string& unique_name,
								     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&gpuSimulateCudaBasebandBeamformer::main_thread, this)) {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_beams = config.get<int>(unique_name, "num_beams");
    voltage_buf = get_buffer("voltage_in_buf");
    phase_buf = get_buffer("phase_in_buf");
    shift_buf = get_buffer("shift_in_buf");
    register_consumer(voltage_buf, unique_name.c_str());
    register_consumer(phase_buf, unique_name.c_str());
    register_consumer(shift_buf, unique_name.c_str());
    output_buf = get_buffer("beams_out_buf");
    register_producer(output_buf, unique_name.c_str());

    /*
      int8_t* cpu_phase_memory = (int8_t*)malloc(phase_len);
      for (size_t i = 0; i < phase_len; i++)
      cpu_phase_memory[i] = 1;

      int32_t* cpu_shift_memory = (int32_t*)malloc(shift_len);
      for (size_t i = 0; i < shift_len / sizeof(int32_t); i++)
      cpu_shift_memory[i] = 1;
    */
}

gpuSimulateCudaBasebandBeamformer::~gpuSimulateCudaBasebandBeamformer() {}

// This code is from Erik's
// https://github.com/eschnett/GPUIndexSpaces.jl/blob/main/kernels/bb.cxx

using int4x2_t = uint8_t;

constexpr int4x2_t set4(const int8_t lo, const int8_t hi) {
  return (uint8_t(lo) & 0x0f) | ((uint8_t(hi) << 4) & 0xf0);
}

constexpr std::array<int8_t, 2> get4(const int4x2_t i) {
  return {int8_t(int8_t((i + 0x08) & 0x0f) - 0x08),
      int8_t(int8_t(((i >> 4) + 0x08) & 0x0f) - 0x08)};
}

void bb_simple(const int8_t *__restrict__ const A,
	       const int4x2_t *__restrict__ const E,
	       //const int8_t *__restrict__ const s,
	       const int32_t *__restrict__ const s,
	       int4x2_t *__restrict__ const J,
	       const int T,   // = 32768; // 32768; // number of times
	       const int B,   // = 96;    // number of beams
	       const int D,   // = 512;   // number of dishes
	       const int F    // = 16;    // frequency channels per GPU
	       ) {
  // J[t,p,f,b] = Σ[d] A[d,b,p,f] E[d,p,f,t]
  //#pragma omp parallel for collapse(2)
  for (int f = 0; f < F; ++f) {
    for (int p = 0; p < 2; ++p) {
      for (int t = 0; t < T; ++t) {
	for (int b = 0; b < B; ++b) {
	  int Jre = 0, Jim = 0;
	  // This pragma slows things down tenfold
	  // #pragma omp simd
	  for (int d = 0; d < D; ++d) {
	    const int Are = A[(((f * 2 * p) * B + b) * D + d) * 2 + 0];
	    const int Aim = A[(((f * 2 * p) * B + b) * D + d) * 2 + 1];
	    const auto [Ere, Eim] = get4(E[((t * F + f) * 2 + p) * D + d]);
	    Jre += Are * Ere - Aim * Eim;
	    Jim += Are * Eim + Aim * Ere;
	  }
	  Jre >>= s[(f * 2 + p) * B + b];
	  Jim >>= s[(f * 2 + p) * B + b];
	  J[((b * F + f) * 2 + p) * T + t] = set4(Jre, Jim);
	}
      }
    }
  }
}

void gpuSimulateCudaBasebandBeamformer::main_thread() {

  int voltage_frame_id = 0;
  int output_frame_id = 0;

  int phase_frame_id = 0;
  int shift_frame_id = 0;

  while (!stop_thread) {
    int4x2_t* voltage = (int4x2_t*)wait_for_full_frame(voltage_buf, unique_name.c_str(), voltage_frame_id);
    if (voltage == nullptr)
      break;
    int8_t* phase = (int8_t*)wait_for_full_frame(phase_buf, unique_name.c_str(), phase_frame_id);
    if (phase == nullptr)
      break;
    int32_t* shift = (int32_t*)wait_for_full_frame(shift_buf, unique_name.c_str(), shift_frame_id);
    if (shift == nullptr)
      break;
    int4x2_t* output = (int4x2_t*)wait_for_empty_frame(output_buf, unique_name.c_str(), output_frame_id);
    if (output == nullptr)
      break;

    INFO("Simulating GPU processing for {:s}[{:d}] {:s}[{:d}] {:s}[{:d}] putting result in {:s}[{:d}]",
	 voltage_buf->buffer_name, voltage_frame_id, phase_buf->buffer_name, phase_frame_id,
	 shift_buf->buffer_name, shift_frame_id, output_buf->buffer_name, output_frame_id);

    /*
      int8_t lo = std::numeric_limits<int8_t>::max();
      int8_t hi = std::numeric_limits<int8_t>::min();
      for (int i=0; i<(_num_elements * _num_local_freq * _num_beams * 2); i++) {
      lo = std::min(lo, phase[i]);
      hi = std::max(hi, phase[i]);
      }
      INFO("Phase range {:d} to {:d}", lo, hi);
    */

    int ndishes = _num_elements / 2;
    bb_simple(phase, voltage, shift, output,
	      _samples_per_data_set, _num_beams, ndishes, _num_local_freq);

    INFO("Simulating GPU processing done for {:s}[{:d}] result is in {:s}[{:d}]",
	 voltage_buf->buffer_name, voltage_frame_id, output_buf->buffer_name, output_frame_id);

    pass_metadata(voltage_buf, voltage_frame_id, output_buf, output_frame_id);
    mark_frame_empty(voltage_buf, unique_name.c_str(), voltage_frame_id);
    mark_frame_full(output_buf, unique_name.c_str(), output_frame_id);

    voltage_frame_id = (voltage_frame_id + 1) % voltage_buf->num_frames;
    output_frame_id = (output_frame_id + 1) % output_buf->num_frames;

    // Check for available phase & shift frames and advance if they're ready!
    int next_frame = (phase_frame_id + 1) % phase_buf->num_frames;
    if (is_frame_empty(phase_buf, next_frame) == 0) {
      mark_frame_empty(phase_buf, unique_name.c_str(), phase_frame_id);
      phase_frame_id = next_frame;
    }
    next_frame = (shift_frame_id + 1) % shift_buf->num_frames;
    if (is_frame_empty(shift_buf, next_frame) == 0) {
      mark_frame_empty(shift_buf, unique_name.c_str(), shift_frame_id);
      shift_frame_id = next_frame;
    }
  }
}
