#include "chordMVPSetup.hpp"

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(chordMVPSetup);

chordMVPSetup::chordMVPSetup(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device) :
    cudaCommand(config, unique_name, host_buffers, device, "chordMVPSetup", "") {
    set_command_type(gpuCommandType::COPY_IN);

    // This stage glues together the cudaUpchannelize and cudaFRBBeamformer commands.
    //   cudaUpchannelize produces outputs of size 2048.
    //   cudaFRBBeamformer consumes inputs of multiples of 48 -- we allocate size 2064.
    // The cudaFRBBeamformer code processes a varying number of input
    // samples each time, so there are some leftovers that it copies
    // into a padding buffer at the beginning of the array.
    ///
    // We allocate the latter, and then create an offset view into it for the former.

    // We also use it to (incorrectly!) take a *time* subset of half
    // the voltage data to feed the Fine Visibility matrix data
    // product.  (We should subset the frequencies, but for MVP
    // purposes we just need to test the rate.  The data layout has T
    // varying most slowly, so we can do a time subset just with array
    // views, while doing the Frequency subset would require a
    // cudaMemcpy3DAsync (probably in this class!).)

    // Finally, we use it to glue FRB1 -> CudaRechunk -> FRB2/3, with
    // a bit of fancy footwork:
    //   FRB1 produces 52 samples, but the last sample is a bit bogus.
    //   FRB2/3 want 256 samples.
    // We construct a buffer view that reduces FRB1's output from 52
    // to 51 for input to CudaRechunk.
    // We then construct a buffer view where CudaRechunk places 255
    // samples of output into a 256-element buffer that is input for
    // FRB2/3.

    // Upchan to FRB-BF:

    size_t num_dishes = config.get<int>(unique_name, "num_dishes");
    // (this is fpga frequencies)
    size_t num_local_freq = config.get<int>(unique_name, "num_local_freq");
    size_t upchan_factor = config.get<int>(unique_name, "upchan_factor");

    // 2048 * U = 32768
    size_t samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    // 2064
    size_t frb_bf_samples = config.get<int>(unique_name, "frb_samples");

    size_t frb_bf_padding = config.get<int>(unique_name, "frb_bf_padding");

    std::string fullname = config.get<std::string>(unique_name, "gpu_mem_frb_bf_input");
    std::string viewname = config.get<std::string>(unique_name, "gpu_mem_upchan_output");
    // 2 for polarizations
    size_t viewsize = num_dishes * num_local_freq * samples_per_data_set * 2;
    size_t fullsize = num_dishes * (num_local_freq * upchan_factor) * (frb_bf_samples + frb_bf_padding) * 2;
    INFO("Creating upchan/frb-bf glue buffers: frb-bf input {:s} size {:d}, upchan output {:s} "
         "size {:d}",
         fullname, fullsize, viewname, viewsize);
    for (int i = 0; i < device.get_gpu_buffer_depth(); i++) {
        void* real = device.get_gpu_memory_array(fullname, i, fullsize);
        // Zero it out!
        CHECK_CUDA_ERROR(cudaMemset((unsigned char*)real + viewsize, 0, fullsize - viewsize));
    }

    size_t offset = num_dishes * num_local_freq * upchan_factor * frb_bf_padding * 2;
    device.create_gpu_memory_array_view(fullname, fullsize, viewname, offset, viewsize);

    // We produce custom DOT output to connect the views, so we omit these (and all other) entries.
    //gpu_buffers_used.push_back(std::make_tuple(fullname, true, false, true));
    //gpu_buffers_used.push_back(std::make_tuple(viewname, true, true, false));

    // FPGA to Upchan for Fine Visibility:

    fullname = config.get<std::string>(unique_name, "gpu_mem_voltage");
    viewname = config.get<std::string>(unique_name, "gpu_mem_fine_upchan_input");

    fullsize = num_dishes * num_local_freq * samples_per_data_set * 2;
    viewsize = fullsize / 2;

    INFO("Creating fpga/fine-upchan glue buffers: fpga {:s} size {:d}, fine-upchan input {:s} "
         "size {:d}",
         fullname, fullsize, viewname, viewsize);
    for (int i = 0; i < device.get_gpu_buffer_depth(); i++)
        device.get_gpu_memory_array(fullname, i, fullsize);
    offset = 0;
    device.create_gpu_memory_array_view(fullname, fullsize, viewname, offset, viewsize);

    // FRB1 writes its output into a view on FRBRechunk's input.

    fullname = config.get<std::string>(unique_name, "gpu_mem_frb_bf_output");
    viewname = config.get<std::string>(unique_name, "gpu_mem_frb_rechunk_input");

    size_t frb_td = config.get<int>(unique_name, "frb_td");
    size_t frb_td_good = config.get<int>(unique_name, "frb_td_good");
    size_t frb_beam_grid_size = config.get<int>(unique_name, "frb_beam_grid_size");
    size_t frb_freq = config.get<int>(unique_name, "frb_freq");
    size_t rho = frb_beam_grid_size * frb_beam_grid_size;

    size_t sizeof_float16_t = 2;

    fullsize = frb_td * frb_freq * rho * sizeof_float16_t;
    viewsize = frb_td_good * frb_freq * rho * sizeof_float16_t;

    INFO("FRB1 to FRBRechunk buffer view: FRB1 output is {:s} size {:d}, Rechunk input {:s} size "
         "{:d}",
         fullname, fullsize, viewname, viewsize);
    for (int i = 0; i < device.get_gpu_buffer_depth(); i++)
        device.get_gpu_memory_array(fullname, i, fullsize);
    offset = 0;
    device.create_gpu_memory_array_view(fullname, fullsize, viewname, offset, viewsize);

    // FRB Rechunk outputs 5 x 51 = 255 into a 256-element array for FRB2

    fullname = config.get<std::string>(unique_name, "gpu_mem_frb_brf_input");
    viewname = config.get<std::string>(unique_name, "gpu_mem_frb_rechunk_output");

    size_t frb_td_rechunk_real = config.get<int>(unique_name, "frb_td_rechunk_real");
    size_t frb_td_rechunk = config.get<int>(unique_name, "frb_td_rechunk");

    fullsize = frb_td_rechunk * frb_freq * rho * sizeof_float16_t;
    viewsize = frb_td_rechunk_real * frb_freq * rho * sizeof_float16_t;

    INFO("FRBRechunk to FRB Beam Reformer buffer view: padded output is {:s} size {:d}, real "
         "output is {:s} size {:d}",
         fullname, fullsize, viewname, viewsize);
    for (int i = 0; i < device.get_gpu_buffer_depth(); i++) {
        void* real = device.get_gpu_memory_array(fullname, i, fullsize);
        // Zero out the last element!
        CHECK_CUDA_ERROR(cudaMemset((unsigned char*)real + viewsize, 0, fullsize - viewsize));
    }
    offset = 0;
    device.create_gpu_memory_array_view(fullname, fullsize, viewname, offset, viewsize);
}

chordMVPSetup::~chordMVPSetup() {}

cudaEvent_t chordMVPSetup::execute(cudaPipelineState& pipestate, const std::vector<cudaEvent_t>& pre_events) {
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
    dot += fmt::format("{:s}\"{:s}\" -> \"{:s}\" [style=solid, color=\"red\"];\n",
                       prefix, fullname, viewname);
    fullname = config.get<std::string>(unique_name, "gpu_mem_frb_bf_output");
    viewname = config.get<std::string>(unique_name, "gpu_mem_frb_rechunk_input");
    dot += fmt::format("{:s}\"{:s}\" -> \"{:s}\" [style=solid, color=\"red\"];\n",
                       prefix, fullname, viewname);

    fullname = config.get<std::string>(unique_name, "gpu_mem_frb_brf_input");
    viewname = config.get<std::string>(unique_name, "gpu_mem_frb_rechunk_output");
    dot += fmt::format("{:s}\"{:s}\" -> \"{:s}\" [style=solid, color=\"red\"];\n",
                       prefix, viewname, fullname);
    return dot;
}
