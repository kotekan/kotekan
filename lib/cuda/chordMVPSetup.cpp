#include "chordMVPSetup.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(chordMVPSetup);

chordMVPSetup::chordMVPSetup(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "chordMVPSetup", "") {
    set_command_type(gpuCommandType::COPY_IN);

    // Upchan to FRB-Beamformer:
    size_t num_dishes = config.get<int>(unique_name, "num_dishes");
    // (this is fpga frequencies)
    size_t num_local_freq = config.get<int>(unique_name, "num_local_freq");
    size_t upchan_factor = config.get<int>(unique_name, "upchan_factor");

    // 2048 * U = 32768
    size_t samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");

    // 2048
    size_t frb_bf_samples = config.get<int>(unique_name, "frb_samples");
    // 88
    size_t frb_bf_padding = config.get<int>(unique_name, "frb_bf_padding");

    std::string fullname = config.get<std::string>(unique_name, "gpu_mem_frb_bf_input");
    std::string viewname = config.get<std::string>(unique_name, "gpu_mem_upchan_output");
    // 2 for polarizations
    size_t viewsize = num_dishes * num_local_freq * samples_per_data_set * 2;
    size_t fullsize =
        num_dishes * (num_local_freq * upchan_factor) * (frb_bf_samples + frb_bf_padding) * 2;
    INFO("Creating upchan/frb-bf glue buffers: frb-bf input {:s} size {:d}, upchan output {:s} "
         "size {:d}",
         fullname, fullsize, viewname, viewsize);

    size_t offset = num_dishes * num_local_freq * upchan_factor * frb_bf_padding * 2;

    for (int i = 0; i < _gpu_buffer_depth; i++) {
        void* real = device.get_gpu_memory_array(fullname, i, _gpu_buffer_depth, fullsize);
        // Zero it out!
        CHECK_CUDA_ERROR(cudaMemset((unsigned char*)real + viewsize, 0, fullsize - viewsize));

        // DEBUG
        DEBUG("GPUMEM memory_array_view({:p}, {:d}, {:p}, {:d}, \"{:s}\", \"{:s}\", {:d}, "
              "\"upchan/frb-bf\")",
              real, fullsize, (char*)real + offset, viewsize, fullname, viewname, i);
    }
    device.create_gpu_memory_array_view(fullname, fullsize, viewname, offset, viewsize, _gpu_buffer_depth);

    // We produce custom DOT output to connect the views, so we omit these (and all other) entries.
    // gpu_buffers_used.push_back(std::make_tuple(fullname, true, false, true));
    // gpu_buffers_used.push_back(std::make_tuple(viewname, true, true, false));

    // Voltage to Upchan for Fine Visibility:

    fullname = config.get<std::string>(unique_name, "gpu_mem_voltage");
    viewname = config.get<std::string>(unique_name, "gpu_mem_fine_upchan_input");

    fullsize = num_dishes * num_local_freq * samples_per_data_set * 2;
    viewsize = fullsize / 2;

    INFO("Creating fpga voltage/fine-upchan glue buffers: fpga {:s} size {:d}, fine-upchan input "
         "{:s} size {:d}",
         fullname, fullsize, viewname, viewsize);
    for (int i = 0; i < _gpu_buffer_depth; i++)
        device.get_gpu_memory_array(fullname, i, _gpu_buffer_depth, fullsize);
    offset = 0;
    device.create_gpu_memory_array_view(fullname, fullsize, viewname, offset, viewsize, _gpu_buffer_depth);
}

chordMVPSetup::~chordMVPSetup() {}

cudaEvent_t chordMVPSetup::execute(cudaPipelineState& pipestate,
                                   const std::vector<cudaEvent_t>& pre_events) {
    (void)pre_events;
    pre_execute(pipestate.gpu_frame_id);
    record_start_event(pipestate.gpu_frame_id);
    return record_end_event(pipestate.gpu_frame_id);
}

std::string chordMVPSetup::get_extra_dot(const std::string& prefix) const {
    std::string fullname = config.get<std::string>(unique_name, "gpu_mem_frb_bf_input");
    std::string viewname = config.get<std::string>(unique_name, "gpu_mem_upchan_output");
    std::string dot = fmt::format("{:s}\"{:s}\" -> \"{:s}\" [style=solid, color=\"red\"];\n",
                                  prefix, viewname, fullname);
    fullname = config.get<std::string>(unique_name, "gpu_mem_voltage");
    viewname = config.get<std::string>(unique_name, "gpu_mem_fine_upchan_input");
    dot += fmt::format("{:s}\"{:s}\" -> \"{:s}\" [style=solid, color=\"red\"];\n", prefix, fullname,
                       viewname);
    return dot;
}
