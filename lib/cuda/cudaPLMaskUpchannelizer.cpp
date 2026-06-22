class cudaPLMaskUpchannelizer : public cudaCommand {
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_expanded_mask;
    NDArrayRingBuffer<kotekan::uint1x8_t, 5> pl_upchannelized_expanded_mask;

public:
    cudaPLMaskUpchannelizer(kotekan::Config& config, const std::string& unique_name,
                            kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                            const int instance_num) :
        pl_expanded_mask(pl_expanded_mask_name, "pl_mask_exp",
                         std::array<std::ptrdiff_t, 5>{
                             buffer_depth * div_noremainder(num_times, 64), num_frequencies,
                             num_polarizations, div_noremainder(num_dishes, 8), 64 / 8},
                         std::array<std::string, 5>{"Thi64", "F", "P", "D8", "Tlo64"}, *this),
        pl_upchannelized_expanded_mask(
            pl_upchannelized_expanded_mask_name, "pl_mask_exp",
            std::array<std::ptrdiff_t, 5>{
                buffer_depth * div_noremainder(num_times, 64 * upchannelization_factor),
                num_frequencies, num_polarizations, div_noremainder(num_dishes, 8), 64 / 8},
            std::array<std::string, 5>{"Thi64", "F", "P", "D8", "Tlo64"}, *this) {}
};
