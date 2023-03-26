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
    //   cudaFRBBeamformer consumer inputs of size 2064.
    // We allocate the latter, and then create a view into it for the former.

    // We also use it to (incorrectly!) take a *time* subset of half
    // the voltage data to feed the Fine Visibility matrix data
    // product.  (We should subset the frequencies, but for MVP
    // purposes we just need to test the rate.  The data layout has T
    // varying most slowly, so we can do a time subset just with array
    // views, while doing the Frequency subset would require a
    // cudaMemcpy3DAsync (probably in this class!).)

    // Upchan to FRB-BF:

    size_t num_dishes = config.get<int>(unique_name, "num_dishes");
    // (this is fpga frequencies)
    size_t num_local_freq = config.get<int>(unique_name, "num_local_freq");
    size_t upchan_factor = config.get<int>(unique_name, "upchan_factor");

    // 2048 * U = 32768
    size_t samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    // 2064
    size_t frb_bf_samples = config.get<int>(unique_name, "frb_samples");

    std::string fullname = config.get<std::string>(unique_name, "gpu_mem_frb_bf_input");
    std::string viewname = config.get<std::string>(unique_name, "gpu_mem_upchan_output");
    // 2 for polarizations
    size_t viewsize = num_dishes * num_local_freq * samples_per_data_set * 2;
    size_t fullsize = num_dishes * (num_local_freq * upchan_factor) * frb_bf_samples * 2;
    INFO("Creating upchan/frb-bf glue buffers: frb-bf input {:s} size {:d}, upchan output {:s} "
         "size {:d}",
         fullname, fullsize, viewname, viewsize);
    for (int i = 0; i < device.get_gpu_buffer_depth(); i++) {
        void* real = device.get_gpu_memory_array(fullname, i, fullsize);
        // Zero it out!
        CHECK_CUDA_ERROR(cudaMemset((unsigned char*)real + viewsize, 0, fullsize - viewsize));
    }
    // offset = 0
    device.create_gpu_memory_array_view(fullname, fullsize, viewname, 0, viewsize);

    // FPGA to Upchan for Fine Visibility:

    fullname = config.get<std::string>(unique_name, "gpu_mem_fpga");
    viewname = config.get<std::string>(unique_name, "gpu_mem_fine_upchan_input");

    fullsize = num_dishes * num_local_freq * samples_per_data_set * 2;
    viewsize = fullsize / 2;

    INFO("Creating fpga/fine-upchan glue buffers: fpga {:s} size {:d}, fine-upchan input {:s} "
         "size {:d}",
         fullname, fullsize, viewname, viewsize);
    for (int i = 0; i < device.get_gpu_buffer_depth(); i++)
        device.get_gpu_memory_array(fullname, i, fullsize);
    // offset = 0
    device.create_gpu_memory_array_view(fullname, fullsize, viewname, 0, viewsize);
}

chordMVPSetup::~chordMVPSetup() {}

cudaEvent_t chordMVPSetup::execute(int gpu_frame_id, const std::vector<cudaEvent_t>& pre_events,
                                   bool* quit) {
    (void)pre_events;
    (void)quit;
    pre_execute(gpu_frame_id);
    record_start_event(gpu_frame_id);
    return record_end_event(gpu_frame_id);
}
