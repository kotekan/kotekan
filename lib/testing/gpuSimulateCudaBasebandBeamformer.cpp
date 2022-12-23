#include "gpuSimulateCudaBasebandBeamformer.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, mark_frame_empty, mark_frame_full, pass_metadata
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO, DEBUG
#include "oneHotMetadata.hpp"

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
}

gpuSimulateCudaBasebandBeamformer::~gpuSimulateCudaBasebandBeamformer() {}

// This code is from Erik's
// https://github.com/eschnett/GPUIndexSpaces.jl/blob/main/kernels/bb.cxx

using int4x2_t = uint8_t;
static constexpr int4x2_t set4(const int8_t lo, const int8_t hi) {
	return (uint8_t(lo) & 0x0f) | ((uint8_t(hi) << 4) & 0xf0);
}

static constexpr std::array<int8_t, 2> get4(const int4x2_t i) {
	return {int8_t(int8_t((i + 0x08) & 0x0f) - 0x08),
			int8_t(int8_t(((i >> 4) + 0x08) & 0x0f) - 0x08)};
}


void gpuSimulateCudaBasebandBeamformer::bb_simple_sub(
												  const int8_t *__restrict__ const A,
												  const int4x2_t *__restrict__ const E,
												  const int32_t *__restrict__ const s,
												  int4x2_t *__restrict__ const J,
												  const int T,   // 32768; // number of times
												  const int B,   // = 96;    // number of beams
												  const int D,   // = 512;   // number of dishes
												  const int F,    // = 16;    // frequency channels per GPU
												  const int t,
												  const int b,
												  const int d,
												  const int f,
												  const int p
													  ) {
	const int f0 = (f == -1 ? 0 : f);
	const int f1 = (f == -1 ? F : f+1);
	const int p0 = (p == -1 ? 0 : p);
	const int p1 = (p == -1 ? 2 : p+1);
	const int b0 = (b == -1 ? 0 : b);
	const int b1 = (b == -1 ? B : b+1);
	const int t0 = (t == -1 ? 0 : t);
	const int t1 = (t == -1 ? T : t+1);
	const int d0 = (d == -1 ? 0 : d);
	const int d1 = (d == -1 ? D : d+1);

	int nprint_v = 0;
	int nprint_b = 0;
	const int nprint_max = 4;

	// J[t,p,f,b] = Σ[d] A[d,b,p,f] E[d,p,f,t]
	//#pragma omp parallel for collapse(2)
	for (int f = f0; f < f1; ++f) {
		//DEBUG("bb_simple frequence channel {:d} of {:d}", f, F);
		for (int p = p0; p < p1; ++p) {
			for (int t = t0; t < t1; ++t) {
				for (int b = b0; b < b1; ++b) {
					int Jre = 0, Jim = 0;
					// This pragma slows things down tenfold
					// #pragma omp simd
					for (int d = d0; d < d1; ++d) {
						//const int Are = A[(((f * 2 + p) * B + b) * D + d) * 2 + 0];
						//const int Aim = A[(((f * 2 + p) * B + b) * D + d) * 2 + 1];
						const int Aim = A[(((f * 2 + p) * B + b) * D + d) * 2 + 0];
						const int Are = A[(((f * 2 + p) * B + b) * D + d) * 2 + 1];
						//const auto [Ere, Eim] = get4(E[((t * 2 + p) * F + f) * D + d]);
						const auto [Eim, Ere] = get4(E[((t * 2 + p) * F + f) * D + d]);
						Jre += Are * Ere - Aim * Eim;
						Jim += Are * Eim + Aim * Ere;
						if ((b == 0) && (Ere || Eim)) {
							size_t indx = ((t * 2 + p) * F + f) * D + d;
							if (nprint_v < nprint_max) {
								DEBUG("bb_simple: found voltage f={:d}, p={:d}, t={:d}, d={:d} = index {:d}=0x{:x} = {:d} = 0x{:x}",
									  f, p, t, d, indx, indx, E[indx], E[indx]);
								nprint_v++;
							}
						}
					}
					int oJre = Jre;
					int oJim = Jim;
					Jre >>= s[(f * 2 + p) * B + b];
					Jim >>= s[(f * 2 + p) * B + b];
					//J[((b * F + f) * 2 + p) * T + t] = set4(Jre, Jim);
					J[((b * F + f) * 2 + p) * T + t] = set4(Jim, Jre);
					if (Jre || Jim) {
						if (nprint_b < nprint_max) {
							DEBUG("bb_simple: setting b={:d}(0x{:x}), f={:d}(0x{:x}), p={:d}(0x{:x}), t={:d}(0x{:x}) = index 0x{:x} = {:d}(0x{:x}); before shift by {:d}, re=0x{:x}, im=0x{:x}",
								  b, b, f, f, p, p, t, t, ((b * F + f) * 2 + p) * T + t, set4(Jim, Jre), set4(Jim,Jre), s[(f * 2 + p) * B + b], oJre, oJim);
							nprint_b++;
						}
					}
				}
			}
		}
	}
}





void gpuSimulateCudaBasebandBeamformer::bb_simple(
												  const int8_t *__restrict__ const A,
												  const int4x2_t *__restrict__ const E,
												  //const int8_t *__restrict__ const s,
												  const int32_t *__restrict__ const s,
												  int4x2_t *__restrict__ const J,
												  const int T,   // = 32768; // 32768; // number of times
												  const int B,   // = 96;    // number of beams
												  const int D,   // = 512;   // number of dishes
												  const int F    // = 16;    // frequency channels per GPU
												  ) {
	bb_simple_sub(A, E, s, J, T, B, D, F, -1, -1, -1, -1, -1);
	/*
	// J[t,p,f,b] = Σ[d] A[d,b,p,f] E[d,p,f,t]
	//#pragma omp parallel for collapse(2)
	for (int f = 0; f < F; ++f) {
		//DEBUG("bb_simple frequence channel {:d} of {:d}", f, F);
		for (int p = 0; p < 2; ++p) {
			for (int t = 0; t < T; ++t) {
				for (int b = 0; b < B; ++b) {
					int Jre = 0, Jim = 0;
					// This pragma slows things down tenfold
					// #pragma omp simd
					for (int d = 0; d < D; ++d) {
						//const int Are = A[(((f * 2 + p) * B + b) * D + d) * 2 + 0];
						//const int Aim = A[(((f * 2 + p) * B + b) * D + d) * 2 + 1];
						const int Aim = A[(((f * 2 + p) * B + b) * D + d) * 2 + 0];
						const int Are = A[(((f * 2 + p) * B + b) * D + d) * 2 + 1];
						//const auto [Ere, Eim] = get4(E[((t * 2 + p) * F + f) * D + d]);
						const auto [Eim, Ere] = get4(E[((t * 2 + p) * F + f) * D + d]);
						Jre += Are * Ere - Aim * Eim;
						Jim += Are * Eim + Aim * Ere;
						if ((b == 0) && (Ere || Eim)) {
							size_t indx = ((t * F + f) * 2 + p) * D + d;
							DEBUG("bb_simple: found voltage f={:d}, p={:d}, t={:d}, d={:d} = index {:d}=0x{:x} = {:d}",
								  f, p, t, d, indx, indx, E[indx]);
						}
					}
					int oJre = Jre;
					int oJim = Jim;
					Jre >>= s[(f * 2 + p) * B + b];
					Jim >>= s[(f * 2 + p) * B + b];
					//J[((b * F + f) * 2 + p) * T + t] = set4(Jre, Jim);
					J[((b * F + f) * 2 + p) * T + t] = set4(Jim, Jre);
					if (Jre || Jim) {
						DEBUG("bb_simple: setting b={:d}(0x{:x}), f={:d}(0x{:x}), p={:d}(0x{:x}), t={:d}(0x{:x}) = index 0x{:x} = {:d}(0x{:x}); before shift by {:d}, re=0x{:x}, im=0x{:x}",
							  b, b, f, f, p, p, t, t, ((b * F + f) * 2 + p) * T + t, set4(Jim, Jre), set4(Jim,Jre), s[(f * 2 + p) * B + b], oJre, oJim);
					}
				}
			}
		}
	}
	*/
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

		bool done = false;
		INFO("Is one-hot? {:}", metadata_is_onehot(voltage_buf, voltage_frame_id));
		if (metadata_is_onehot(voltage_buf, voltage_frame_id)) {
			INFO("One-hot voltage input matrix, running quick version");
			std::vector<int> inds = get_onehot_indices(voltage_buf, voltage_frame_id);
			if (inds.size() != 4) {
				INFO("Expected 4 indices in one-hot array, got {:d}", inds.size());
			} else {
				int t = inds[0];
				int p = inds[1];
				int f = inds[2];
				int d = inds[3];
				int b = -1;
				INFO("One-hot: time {:d} pol {:d}, freq {:d}, dish {:d}", t, p, f, d);
				int ndishes = _num_elements / 2;
				bb_simple_sub(phase, voltage, shift, output,
							  _samples_per_data_set, _num_beams, ndishes, _num_local_freq,
							  t, b, d, f, p);
				done = true;
			}
		}

		if (!done) {
			int ndishes = _num_elements / 2;
			bb_simple(phase, voltage, shift, output,
					  _samples_per_data_set, _num_beams, ndishes, _num_local_freq);
		}

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
