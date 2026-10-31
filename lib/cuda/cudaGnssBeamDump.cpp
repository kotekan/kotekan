#include "cudaGnssBeamDump.hpp"

#include "cudaUtils.hpp" // for CHECK_CUDA_ERROR
#include "kotekanLogging.hpp"

#include <cassert>
#include <chordMetadata.hpp>
#include <cstring>

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaGnssBeamDump);

cudaGnssBeamDump::cudaGnssBeamDump(Config& config, const std::string& unique_name,
                                   bufferContainer& host_buffers, cudaDeviceInterface& device,
                                   int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num),
    _buffer_depth(config.get<int>(unique_name, "buffer_depth")),
    _num_times(config.get<int>(unique_name, "num_times")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    _beam_name(config.get<std::string>(unique_name, "beam_name")),
    _local_channels(config.get<std::vector<std::int32_t>>(unique_name, "beam_local_channels")),
    _freq_ids(config.get<std::vector<std::int32_t>>(unique_name, "beam_freq_ids")),
    _ref_input(config.get<int>(unique_name, "beam_ref_input")),
    _eig_iters(config.get_default<int>(unique_name, "beam_eig_iters", 4)),
    _layout(gnss_cuda::beam_dump_layout((int)_local_channels.size(), _num_times)),
    voltage(_voltage_name, "E",
            std::array<std::ptrdiff_t, 4>{_buffer_depth * _num_times, _num_local_freq, 2,
                                          _num_elements / 2},
            std::array<std::string, 4>{"T", "F", "P", "D"},
            std::array<std::ptrdiff_t, 4>{1, 1, 1, 1}, *this),
    beam(_beam_name, "gnss_beam", std::array<std::ptrdiff_t, 2>{1, (std::ptrdiff_t)_layout.bytes},
         std::array<std::string, 2>{"B", "X"}, std::array<std::ptrdiff_t, 2>{1, 1}, *this) {
    if (_num_elements != gnss_cuda::BEAM_INPUTS)
        FATAL_ERROR("cudaGnssBeamDump: num_elements {:d} != {:d}; the kernels are written for "
                    "128 inputs",
                    _num_elements, gnss_cuda::BEAM_INPUTS);
    if (_num_times % 256)
        FATAL_ERROR("cudaGnssBeamDump: num_times {:d} must be a multiple of 256", _num_times);
    if (_local_channels.empty() || _local_channels.size() != _freq_ids.size())
        FATAL_ERROR("cudaGnssBeamDump: beam_local_channels ({:d}) and beam_freq_ids ({:d}) must "
                    "be non-empty and the same length",
                    (int)_local_channels.size(), (int)_freq_ids.size());
    for (std::int32_t f : _local_channels)
        if (f < 0 || f >= _num_local_freq)
            FATAL_ERROR("cudaGnssBeamDump: beam_local_channels entry {:d} outside [0, {:d}) -- "
                        "LOCAL indices, not global freq_ids",
                        f, _num_local_freq);
    if (_ref_input < 0 || _ref_input >= _num_elements)
        FATAL_ERROR("cudaGnssBeamDump: beam_ref_input {:d} outside [0, {:d})", _ref_input,
                    _num_elements);
    if (_eig_iters < 1 || _eig_iters > 64)
        FATAL_ERROR("cudaGnssBeamDump: beam_eig_iters {:d} outside [1, 64]", _eig_iters);

    voltage.register_consumer();
    gpu_buffers_used.push_back(std::make_tuple(_beam_name, true, false, true));

    const int n = (int)_local_channels.size();
    // Shared scratch and the warm-start state, named here so --dry-run sees them. The state
    // is zeroed once (a cold start); every instance's constructor repeats the idempotent fill.
    const size_t cov_bytes =
        (size_t)n * gnss_cuda::BEAM_INPUTS * gnss_cuda::BEAM_INPUTS * sizeof(float2);
    const size_t q_bytes = (size_t)n * gnss_cuda::BEAM_INPUTS * sizeof(float2);
    device.get_gpu_memory(unique_name + "_cov", cov_bytes);
    void* d_q1 = device.get_gpu_memory(unique_name + "_q1", q_bytes);
    void* d_q2 = device.get_gpu_memory(unique_name + "_q2", q_bytes);
    void* d_sel = device.get_gpu_memory(unique_name + "_sel", (size_t)n * sizeof(int));
    if (instance_num == 0) {
        CHECK_CUDA_ERROR(cudaMemset(d_q1, 0, q_bytes));
        CHECK_CUDA_ERROR(cudaMemset(d_q2, 0, q_bytes));
        CHECK_CUDA_ERROR(cudaMemcpy(d_sel, _local_channels.data(), (size_t)n * sizeof(int),
                                    cudaMemcpyHostToDevice));
    }

    _hdr_stage.assign(_layout.off_freq + sizeof(std::int32_t) * n, 0);
    auto* hdr = reinterpret_cast<gnss_cuda::BeamDumpHdr*>(_hdr_stage.data());
    hdr->magic = gnss_cuda::BEAM_DUMP_MAGIC;
    hdr->version = gnss_cuda::BEAM_DUMP_VERSION;
    hdr->n_chan = n;
    hdr->n_times = _num_times;
    hdr->n_inputs = _num_elements;
    hdr->ref_input = _ref_input;
    hdr->conj = 0;
    hdr->eig_iters = _eig_iters;
    std::memcpy(_hdr_stage.data() + _layout.off_freq, _freq_ids.data(), sizeof(std::int32_t) * n);

    set_command_type(gpuCommandType::KERNEL);
    INFO("cudaGnssBeamDump: {:d} channels (freq_ids {:d}..{:d}) x {:d} samples into '{:s}' "
         "({:.2f} MB/frame), reference input {:d}, {:d} power iterations",
         n, _freq_ids.front(), _freq_ids.back(), _num_times, _beam_name, _layout.bytes / 1.0e6,
         _ref_input, _eig_iters);
}

cudaGnssBeamDump::~cudaGnssBeamDump() {}

int cudaGnssBeamDump::wait_on_precondition() {
    const int errcode = voltage.wait_and_claim_readable([&](const std::ptrdiff_t available) {
        if (available < _num_times)
            return read_descriptor_t{.claimed = 0, .read = 0};
        else
            return read_descriptor_t{.claimed = _num_times, .read = _num_times};
    });
    return errcode < 0 ? errcode : 0;
}

cudaEvent_t cudaGnssBeamDump::execute(cudaPipelineState& pipestate,
                                      const std::vector<cudaEvent_t>&) {
    (void)pipestate;
    pre_execute();
    record_start_event();
    const cudaStream_t stream = device.getStream(cuda_stream_id);
    const int n = (int)_local_channels.size();

    voltage.check_metadata();
    const std::shared_ptr<const chordMetadata> meta = voltage.get_metadata();
    if (!_checked_freqs) {
        // The config's freq_ids were resolved at generation time; the ring's labels are the
        // authority. A mismatch means a rewired comb, and a beam on the wrong channel is worse
        // than none.
        const std::vector<int> coarse = meta->get_coarse_freq();
        for (int c = 0; c < n; c++) {
            const int f = _local_channels[(size_t)c];
            if (f >= (int)coarse.size() || coarse[(size_t)f] != _freq_ids[(size_t)c])
                FATAL_ERROR("cudaGnssBeamDump: local channel {:d} carries freq_id {:d}, config "
                            "expects {:d}",
                            f, f < (int)coarse.size() ? coarse[(size_t)f] : -1,
                            _freq_ids[(size_t)c]);
        }
        _checked_freqs = true;
    }
    const long long seq0 = meta->get_fpga_seq_num()
                           + voltage.get_read_valid().begin() * meta->get_time_downsampling_fpga();
    beam.set_metadata(meta);
    beam.get_metadata()->set_fpga_seq_num(seq0);

    const std::ptrdiff_t time_offset =
        voltage.get_read_valid().begin() % voltage.get_ndarray().extent(0);
    assert((voltage.get_read_valid().end() - 1) % voltage.get_ndarray().extent(0) >= time_offset);
    const auto* d_in =
        reinterpret_cast<const unsigned char*>(&voltage.get_ndarray()(time_offset, 0, 0, 0));
    unsigned char* d_out = beam.get_ndarray().data();

    const size_t cov_bytes =
        (size_t)n * gnss_cuda::BEAM_INPUTS * gnss_cuda::BEAM_INPUTS * sizeof(float2);
    const size_t q_bytes = (size_t)n * gnss_cuda::BEAM_INPUTS * sizeof(float2);
    auto* d_cov = (float2*)device.get_gpu_memory(unique_name + "_cov", cov_bytes);
    auto* d_q1 = (float2*)device.get_gpu_memory(unique_name + "_q1", q_bytes);
    auto* d_q2 = (float2*)device.get_gpu_memory(unique_name + "_q2", q_bytes);
    auto* d_sel = (const int*)device.get_gpu_memory(unique_name + "_sel", (size_t)n * sizeof(int));

    // Header + freq_ids. The staging is pageable, so this copy returns once the bytes are
    // taken, and the vector is reusable next frame.
    auto* hdr = reinterpret_cast<gnss_cuda::BeamDumpHdr*>(_hdr_stage.data());
    hdr->seq0 = seq0;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_out, _hdr_stage.data(), _hdr_stage.size(),
                                     cudaMemcpyHostToDevice, stream));

    gnss_cuda::launch_beam_cov(d_in, d_sel, n, _num_local_freq, _num_times, d_cov, stream);
    gnss_cuda::launch_beam_eig(d_cov, n, _num_times, _ref_input, _eig_iters, d_q1, d_q2,
                               (float2*)(d_out + _layout.off_w), (float*)(d_out + _layout.off_pw),
                               (float*)(d_out + _layout.off_lam), stream);
    gnss_cuda::launch_beam_form(d_in, d_sel, n, _num_local_freq, _num_times,
                                (const float2*)(d_out + _layout.off_w), d_out + _layout.off_beam,
                                stream);
    if ((++_frames & 0x3FF) == 1)
        INFO("cudaGnssBeamDump: frame seq0 {:d}, {:d} channels", seq0, n);
    return record_end_event();
}

void cudaGnssBeamDump::finalize_frame() {
    voltage.finish_read();
    cudaCommand::finalize_frame();
}
