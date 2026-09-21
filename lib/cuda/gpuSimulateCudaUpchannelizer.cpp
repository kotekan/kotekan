#include "Config.hpp"                // for Config
#include "DataType.hpp"              // for int4x2_swapped_withoffset_t, float16_t, cint8
#include "NDArrayBuffer.hpp"         // for NDArrayBuffer, buffer_type_t
#include "NDArrayRingBuffer.hpp"     // for NDArrayRingBuffer, read_descriptor_t
#include "bufferContainer.hpp"       // for bufferContainer
#include "chordMetadata.hpp"         // for chordMetadata
#include "cudaCommand.hpp"           // for cudaCommand, REGISTER_CUDA_COMMAND
#include "cudaDeviceInterface.hpp"   // for cudaDeviceInterface
#include "cudaUtils.hpp"             // for CHECK_CUDA_ERROR
#include "div.hpp"                   // for div_noremainder, round_down
#include "kotekanLogging.hpp"        // for FATAL_ERROR, INFO
#include "upchannelizeReference.hpp" // for upchannelize_reference, upchan_window

#include <algorithm>          // for min
#include <array>              // for array
#include <cassert>            // for assert
#include <complex>            // for complex
#include <cstddef>            // for ptrdiff_t, size_t
#include <cstdint>            // for int8_t, int64_t
#include <cuda_runtime_api.h> // for cudaMemcpy, cudaStreamSynchronize
#include <string>             // for string
#include <type_traits>        // for is_same_v
#include <vector>             // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::div_noremainder, kotekan::round_down;

/**
 * @class gpuSimulateCudaUpchannelizer
 * @brief Serial CPU reference for the generated @c cudaUpchannelizer_<setup>_U<N> kernels.
 *
 * A correct-but-slow drop-in replacement: it consumes and produces the same GPU ring buffers
 * as the real kernel, so a config can swap the two by changing one @c name: line. The voltage
 * data is copied off the device, upchannelized serially in float32 on the host, and copied
 * back.
 *
 * The transform is eqn. (83) of Kendrick Smith's notes,
 * @c julia/docs/CHORD_GPU_upchannelization.pdf, implemented literally in
 * @c lib/cuda/upchannelizeReference.cpp -- no FFT, no tensor cores, no float16. Comparing
 * this against the generated kernel is the only check that the kernel computes what the notes
 * specify; see @c config/ci-tests/gpu_batch/verify_cuda_upchan.j2.
 *
 * Unlike the generated commands, whose array shapes are compile-time constants, this one takes
 * every shape from config, so a single class covers every setup and every upchannelization
 * factor.
 *
 * @warning This is thousands of times slower than the real kernel. Use it on small test
 *          configs, not on a production pipeline.
 *
 * @warning Run it with @c buffer_depth: 1 on its enclosing @c cudaProcess. All instances of a
 *          command share one ring-buffer read cursor, and @c RingBuffer::finish_read advances
 *          the shared tail on whichever instance finishes first. For a kernel that takes
 *          microseconds the instances complete in order and that is harmless; this one takes
 *          seconds per call, so several instances overlap, finish out of order, and let the
 *          producer overwrite input another instance is still reading. Pipelining buys nothing
 *          here anyway.
 *
 * @par GPU Memory
 * @gpu_mem Input voltages
 *   @gpu_mem_buffer       @c ring
 *   @gpu_mem_quantity     @c E
 *   @gpu_mem_type         @c int4x2_swapped_withoffset
 *   @gpu_mem_dim_name     [@c T][@c F][@c P][@c D]
 *   @gpu_mem_metadata     @c chordMetadata
 * @gpu_mem Upchannelization gains
 *   @gpu_mem_buffer       @c ndarray
 *   @gpu_mem_quantity     @c G
 *   @gpu_mem_type         @c float16
 *   @gpu_mem_dim_name     [@c Fbar]
 *   @gpu_mem_metadata     @c chordMetadata
 * @gpu_mem Output upchannelized voltages
 *   @gpu_mem_buffer       @c ring
 *   @gpu_mem_quantity     @c Ebar
 *   @gpu_mem_type         @c int4x2_swapped_withoffset (or @c cint8 for the @c _K8 variant)
 *   @gpu_mem_dim_name     [@c Tbar][@c Fbar][@c P][@c D]
 *   @gpu_mem_metadata     @c chordMetadata
 *
 * @conf  num_dishes            Int.  Number of dishes @f$D@f$.
 * @conf  num_polarizations     Int.  Number of polarizations @f$P@f$.
 * @conf  num_frequencies       Int.  Number of input coarse channels @f$F@f$.
 * @conf  num_frequencies_out   Int.  Number of output fine channels the buffer holds,
 *                                    i.e. @c upchan_max_num_channels * @c U.
 * @conf  ring_num_times        Int.  Length of the input voltage ring along time, i.e. the
 *                                    generated kernel's @c cuda_max_number_of_timesamples. The
 *                                    output ring is @c ring_num_times / @c U long. This is
 *                                    deliberately *not* derived from @c buffer_depth, so that
 *                                    the enclosing @c cudaProcess can be given
 *                                    @c buffer_depth: 1 -- see the warning below.
 * @conf  upchannelization_factor  Int.  @f$U@f$.
 * @conf  num_taps              Int.  PFB taps @f$M@f$, default 4.
 * @conf  granularity_number_of_timesamples  Int.  Time samples processed per call, default
 *                                    256. Affects only chunking, not the result.
 * @conf  max_times_per_iteration  Int.  Cap on the input samples read per call, default 0
 *                                    (meaning a quarter of the ring, as the real kernel uses).
 *                                    Lower it to bound this stage's host memory use.
 * @conf  Fmin                  Int.  First input coarse channel to upchannelize.
 * @conf  Fmax                  Int.  One past the last input coarse channel.
 * @conf  voltage_name          String.  Base name for the input voltage buffers.
 * @conf  upchan_gain_name      String.  Base name for the gain buffer.
 * @conf  upchan_voltage_name   String.  Base name for the output voltage buffers.
 */
template<typename OutT>
class gpuSimulateCudaUpchannelizerT : public cudaCommand {
public:
    gpuSimulateCudaUpchannelizerT(Config& config, const std::string& unique_name,
                                  bufferContainer& host_buffers, cudaDeviceInterface& device,
                                  int instance_num);
    virtual ~gpuSimulateCudaUpchannelizerT();

    int wait_on_precondition() override;
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;
    void finalize_frame() override;

private:
    // How many input samples we can turn into output. The overlap is the (M-1)*U samples that
    // every call must re-read, because the output at `tbar` reaches M*U samples forward.
    std::int64_t num_processed_elements(std::int64_t available) const {
        return round_down(available, std::int64_t(_granularity));
    }
    std::int64_t num_consumed_elements(std::int64_t available) const {
        const std::int64_t processed = num_processed_elements(available);
        const std::int64_t overlap = std::int64_t(_upchannelization_factor) * (_num_taps - 1);
        return processed < overlap ? 0 : processed - overlap;
    }
    std::int64_t num_produced_elements(std::int64_t available) const {
        return div_noremainder(num_consumed_elements(available),
                               std::int64_t(_upchannelization_factor));
    }

    /// Fill the output metadata and validate the shapes, once, on the first frame.
    void set_metadata_once();

    const int _num_dishes;
    const int _num_polarizations;
    const int _num_frequencies;
    const int _num_frequencies_out;
    const int _ring_num_times;
    const int _upchannelization_factor;
    const int _num_taps;
    const int _granularity;
    const int _max_times_per_iteration;
    const int _Fmin;
    const int _Fmax;

    const std::string _gain_name;
    const std::string _voltage_name;
    const std::string _upchan_voltage_name;

    NDArrayBuffer<float16_t, 1> G_buffer;
    NDArrayRingBuffer<kotekan::int4x2_swapped_withoffset_t, 4> E_buffer;
    NDArrayRingBuffer<OutT, 4> Ebar_buffer;

    /// The PFB window, W(s), for s in [0, M*U). Computed once, after the shape checks.
    std::vector<float> _window;
    /// The gains, one per output fine channel. Copied off the device on the first frame.
    std::vector<float> _gain;

    /// Set once, on the first frame; see `NDArrayRingBuffer::set_metadata`.
    bool did_set_metadata;
};

////////////////////////////////////////////////////////////////////////////////

template<typename OutT>
gpuSimulateCudaUpchannelizerT<OutT>::gpuSimulateCudaUpchannelizerT(Config& config,
                                                                   const std::string& unique_name,
                                                                   bufferContainer& host_buffers,
                                                                   cudaDeviceInterface& device,
                                                                   const int instance_num) :
    cudaCommand(config, unique_name, host_buffers, device, instance_num),

    _num_dishes(config.get<int>(unique_name, "num_dishes")),
    _num_polarizations(config.get<int>(unique_name, "num_polarizations")),
    _num_frequencies(config.get<int>(unique_name, "num_frequencies")),
    _num_frequencies_out(config.get<int>(unique_name, "num_frequencies_out")),
    _ring_num_times(config.get<int>(unique_name, "ring_num_times")),
    _upchannelization_factor(config.get<int>(unique_name, "upchannelization_factor")),
    _num_taps(config.get_default<int>(unique_name, "num_taps", kotekan::upchan_default_num_taps)),
    _granularity(config.get_default<int>(unique_name, "granularity_number_of_timesamples", 256)),
    _max_times_per_iteration(config.get_default<int>(unique_name, "max_times_per_iteration", 0)),
    _Fmin(config.get<int>(unique_name, "Fmin")), _Fmax(config.get<int>(unique_name, "Fmax")),

    _gain_name(config.get<std::string>(unique_name, "upchan_gain_name")),
    _voltage_name(config.get<std::string>(unique_name, "voltage_name")),
    _upchan_voltage_name(config.get<std::string>(unique_name, "upchan_voltage_name")),

    G_buffer(_gain_name, "G", std::array<std::ptrdiff_t, 1>{_num_frequencies_out},
             std::array<std::string, 1>{"Fbar"}, std::array<std::ptrdiff_t, 1>{1}, *this,
             buffer_type_t::do_once),
    E_buffer(_voltage_name, "E",
             std::array<std::ptrdiff_t, 4>{_ring_num_times, _num_frequencies, _num_polarizations,
                                           _num_dishes},
             std::array<std::string, 4>{"T", "F", "P", "D"},
             std::array<std::ptrdiff_t, 4>{1, 1, 1, 1}, *this),
    Ebar_buffer(
        _upchan_voltage_name, "Ebar",
        std::array<std::ptrdiff_t, 4>{div_noremainder(std::ptrdiff_t(_ring_num_times),
                                                      std::ptrdiff_t(_upchannelization_factor)),
                                      _num_frequencies_out, _num_polarizations, _num_dishes},
        std::array<std::string, 4>{"Tbar", "Fbar", "P", "D"},
        std::array<std::ptrdiff_t, 4>{_upchannelization_factor, 1, 1, 1}, *this),

    did_set_metadata(false) {

    if (!(_upchannelization_factor >= 1))
        FATAL_ERROR("upchannelization_factor must be positive, not {:d}", _upchannelization_factor);
    if (!(_num_taps >= 1))
        FATAL_ERROR("num_taps must be positive, not {:d}", _num_taps);
    // The ring bookkeeping subtracts a whole number of output samples as overlap.
    if (_granularity % _upchannelization_factor != 0)
        FATAL_ERROR("granularity_number_of_timesamples ({:d}) must be a multiple of the "
                    "upchannelization factor ({:d})",
                    _granularity, _upchannelization_factor);
    if (!(0 <= _Fmin && _Fmin <= _Fmax && _Fmax <= _num_frequencies))
        FATAL_ERROR("Invalid frequency span [{:d},{:d}): there are {:d} input frequencies", _Fmin,
                    _Fmax, _num_frequencies);
    // Each input channel becomes U output channels, and they have to fit.
    if (_upchannelization_factor * (_Fmax - _Fmin) > _num_frequencies_out)
        FATAL_ERROR("Upchannelizing [{:d},{:d}) by {:d} produces {:d} frequencies, but the "
                    "output buffer holds only {:d}",
                    _Fmin, _Fmax, _upchannelization_factor,
                    _upchannelization_factor * (_Fmax - _Fmin), _num_frequencies_out);

    _window = kotekan::upchan_window(_num_taps, _upchannelization_factor);

    G_buffer.register_consumer();
    E_buffer.register_consumer();
    Ebar_buffer.register_producer();

    set_command_type(gpuCommandType::KERNEL);

    // One line per command, not one per instance.
    if (instance_num == 0)
        INFO("Serial CPU upchannelizer: U={:d} M={:d}, coarse channels [{:d},{:d}) of {:d}, "
             "{:d} dishes x {:d} polarizations. This is a validation reference and is very "
             "slow.",
             _upchannelization_factor, _num_taps, _Fmin, _Fmax, _num_frequencies, _num_dishes,
             _num_polarizations);
}

template<typename OutT>
gpuSimulateCudaUpchannelizerT<OutT>::~gpuSimulateCudaUpchannelizerT() {}

template<typename OutT>
int gpuSimulateCudaUpchannelizerT<OutT>::wait_on_precondition() {
    {
        const int errcode = cudaCommand::wait_on_precondition();
        if (errcode < 0)
            return errcode;
    }

    // Wait for data to be available in the input ring buffer. This mirrors the generated
    // kernel exactly: the two must claim and release the same elements, or swapping one for
    // the other in a config would change where the output lands.
    const std::ptrdiff_t T_ringbuf = E_buffer.get_ndarray().extent(0);
    const std::ptrdiff_t T_read_max =
        _max_times_per_iteration > 0
            ? std::min<std::ptrdiff_t>(_max_times_per_iteration, T_ringbuf / 4)
            : T_ringbuf / 4;
    std::ptrdiff_t T_read = -1;
    {
        const int errcode = E_buffer.wait_and_claim_readable([&](const std::ptrdiff_t T_available) {
            using std::min;
            T_read = min(T_available, T_read_max);
            // Ensure that we make progress: if we cannot claim any elements then we must not
            // read any either, and instead wait for more data.
            const std::ptrdiff_t T_claimed = num_consumed_elements(T_read);
            const std::ptrdiff_t T_processed = T_claimed == 0 ? 0 : num_processed_elements(T_read);
            return read_descriptor_t{.claimed = T_claimed, .read = T_processed};
        });
        if (errcode < 0)
            return errcode;
    }

    // Wait for space in the output ring buffer
    {
        const int errcode = Ebar_buffer.wait_for_writable(num_produced_elements(T_read));
        if (errcode < 0)
            return errcode;
    }

    return 0;
}

template<typename OutT>
void gpuSimulateCudaUpchannelizerT<OutT>::set_metadata_once() {
    const int U = _upchannelization_factor;

    G_buffer.check_metadata();
    E_buffer.check_metadata();
    Ebar_buffer.set_metadata(E_buffer.get_metadata());

    const auto E_meta = E_buffer.get_metadata();
    auto Ebar_meta = Ebar_buffer.get_metadata();

    const auto E_nfreq = E_meta->get_nfreq();
    if (!(0 <= _Fmin && _Fmin <= _Fmax && _Fmax <= E_nfreq))
        FATAL_ERROR("Invalid frequency span [{:d},{:d}): input buffer E holds {:d} frequencies",
                    _Fmin, _Fmax, E_nfreq);
    const int Ebar_nfreq = U * (_Fmax - _Fmin);

    // Each output fine channel inherits its coarse channel's upchannelization, times U.
    const auto E_freq_upchan_factor = E_meta->get_freq_upchan_factor();
    std::vector<int> Ebar_freq_upchan_factor(Ebar_nfreq);
    std::vector<int> Ebar_freq_upchan_index(Ebar_nfreq);
    std::vector<int> Ebar_coarse_freq(Ebar_nfreq);
    const auto E_coarse_freq = E_meta->get_coarse_freq();
    for (int freq = 0; freq < Ebar_nfreq; ++freq) {
        const int coarse_freq = _Fmin + freq / U;
        assert(coarse_freq < _Fmax);
        Ebar_freq_upchan_factor.at(freq) = E_freq_upchan_factor.at(coarse_freq) * U;
        Ebar_freq_upchan_index.at(freq) = freq % U;
        Ebar_coarse_freq.at(freq) = E_coarse_freq.at(coarse_freq);
    }
    Ebar_meta->set_freq_upchan_factor(Ebar_freq_upchan_factor);
    Ebar_meta->set_freq_upchan_index(Ebar_freq_upchan_index);
    Ebar_meta->set_coarse_freq(Ebar_coarse_freq);

    Ebar_meta->set_time_downsampling_fpga(E_meta->get_time_downsampling_fpga() * U);

    // Mismatched gains would scale each frequency by another frequency's gain.
    const auto G_meta = G_buffer.get_metadata();
    const auto G_nfreq = G_meta->get_nfreq();
    if (G_nfreq != Ebar_nfreq)
        FATAL_ERROR("Gain buffer G holds {:d} frequencies, but this kernel produces {:d}", G_nfreq,
                    Ebar_nfreq);
    const auto G_coarse_freq = G_meta->get_coarse_freq();
    for (int freq = 0; freq < Ebar_nfreq; ++freq)
        if (Ebar_coarse_freq.at(freq) != G_coarse_freq.at(freq))
            FATAL_ERROR("Gain buffer G is for coarse frequency {:d} at index {:d}, but this "
                        "kernel produces coarse frequency {:d} there",
                        G_coarse_freq.at(freq), freq, Ebar_coarse_freq.at(freq));

    // The buffers must be large enough for the frequencies we are about to read and write.
    if (E_meta->dim[1] != E_nfreq)
        FATAL_ERROR("Input buffer E reports {:d} frequencies, but its frequency dimension has "
                    "extent {:d}",
                    E_nfreq, E_meta->dim[1]);
    if (Ebar_meta->dim[1] < Ebar_nfreq)
        FATAL_ERROR("This kernel produces {:d} frequencies, but the frequency dimension of its "
                    "output buffer Ebar has extent {:d}",
                    Ebar_nfreq, Ebar_meta->dim[1]);

    // Since we use a ring buffer we do not need to update `meta->fpga_seq_num`.
}

template<typename OutT>
cudaEvent_t gpuSimulateCudaUpchannelizerT<OutT>::execute(cudaPipelineState& /*pipestate*/,
                                                         const std::vector<cudaEvent_t>& /*pre*/) {
    pre_execute();
    record_start_event();

    const int U = _upchannelization_factor;
    const int M = _num_taps;
    const int P = _num_polarizations;
    const int D = _num_dishes;
    const int F_length = _Fmax - _Fmin;
    const int Fbar_length = U * F_length;
    // Number of values per (time, coarse frequency): the P and D axes are the innermost two.
    const std::ptrdiff_t PD = std::ptrdiff_t(P) * D;

    // Everything this command reads from the device -- the gains and the input voltages -- is
    // written by commands that only *enqueue* their work into our stream (cudaInputData,
    // cudaSyncInput, the upstream ring producer). None of it has necessarily completed when
    // execute() runs on the host, so drain the stream before the first device read. Doing this
    // after the gain copy instead lets a later instance pick up the gains before
    // `cudaInputData` has uploaded them, which silently corrupts that instance's whole output
    // block.
    CHECK_CUDA_ERROR(cudaStreamSynchronize(device.getStream(cuda_stream_id)));

    // Only instance 0 publishes the output metadata; see `NDArrayRingBuffer::set_metadata`.
    if (instance_num == 0 && !did_set_metadata) {
        did_set_metadata = true;
        set_metadata_once();
    }

    // Every instance needs its own host copy of the gains: unlike the real kernel, which reads
    // them from device memory, we apply them on the CPU. They are `do_once`, so one copy per
    // instance covers the whole run.
    if (_gain.empty()) {
        const float16_t* const G_device = G_buffer.get_ndarray().data();
        std::vector<float16_t> G_host(_num_frequencies_out);
        CHECK_CUDA_ERROR(cudaMemcpy(G_host.data(), G_device, G_host.size() * sizeof *G_host.data(),
                                    cudaMemcpyDeviceToHost));
        _gain.resize(_num_frequencies_out);
        for (int freq = 0; freq < _num_frequencies_out; ++freq)
            _gain.at(freq) = float(G_host.at(freq));
    }

    if (!Ebar_buffer.has_metadata())
        FATAL_ERROR("Output buffer Ebar has no metadata; the CPU upchannelizer cannot run");

    const std::ptrdiff_t T_min = E_buffer.get_read_valid().begin();
    const std::ptrdiff_t T_max = E_buffer.get_read_valid().end();
    const std::ptrdiff_t Tbar_min = Ebar_buffer.get_write_valid().begin();
    const std::ptrdiff_t Tbar_max = Ebar_buffer.get_write_valid().end();
    const std::ptrdiff_t T_length = T_max - T_min;
    const std::ptrdiff_t Tbar_length = Tbar_max - Tbar_min;

    // Output `Tbar_min + j` reads inputs [T_min + U*j, T_min + U*j + M*U). Two invariants make
    // that well defined, and both are cheap to state, so state them rather than assume them: a
    // wrong alignment would silently shift every output sample by a few coarse times, which is
    // exactly the kind of error this stage exists to catch in the real kernel.
    //
    // First, the input and output heads advance together: every call claims a whole number of
    // output samples' worth of input, so T_min == U * Tbar_min holds from the very first call.
    if (T_min != std::ptrdiff_t(U) * Tbar_min)
        FATAL_ERROR("Input ring is at sample {:d} but the output ring is at {:d}, which with "
                    "U={:d} should correspond to input sample {:d}",
                    T_min, Tbar_min, U, std::ptrdiff_t(U) * Tbar_min);
    // Second, the last output reaches exactly T_max, so we neither read past the claimed span
    // nor leave part of it unused.
    if (std::ptrdiff_t(U) * (Tbar_length - 1) + std::ptrdiff_t(M) * U != T_length)
        FATAL_ERROR("Reading {:d} input samples cannot produce {:d} output samples with U={:d}, "
                    "M={:d}",
                    T_length, Tbar_length, U, M);

    // Bring the input off the device: [T_length][F_length][P][D], packed.
    const std::vector<kotekan::int4x2_swapped_withoffset_t> E_host =
        E_buffer.copy_to_host(_Fmin, _Fmax, T_min, T_max);

    std::vector<OutT> Ebar_host(std::size_t(Tbar_length) * Fbar_length * PD);

    // Scratch for one coarse channel at a time: the reference wants its input and output
    // contiguous, and the ring layout interleaves the frequency axis between time and dish.
    std::vector<std::complex<float>> E_channel(std::size_t(T_length) * PD);
    std::vector<std::complex<float>> Ebar_channel(std::size_t(Tbar_length) * U * PD);

    for (int fi = 0; fi < F_length; ++fi) {
        // Decode this coarse channel into float32.
        for (std::ptrdiff_t t = 0; t < T_length; ++t) {
            const kotekan::int4x2_swapped_withoffset_t* const in =
                E_host.data() + t * F_length * PD + std::ptrdiff_t(fi) * PD;
            std::complex<float>* const out = E_channel.data() + t * PD;
            for (std::ptrdiff_t e = 0; e < PD; ++e)
                out[e] = kotekan::upchan_decode_int4(in[e].val);
        }

        kotekan::upchannelize_reference(E_channel.data(), _window.data(),
                                        _gain.data() + std::size_t(fi) * U, Ebar_channel.data(), M,
                                        U, P, D, int(Tbar_length));

        // Encode into the output, whose fine channels for this coarse channel are
        // [U*fi, U*(fi+1)).
        for (std::ptrdiff_t tbar = 0; tbar < Tbar_length; ++tbar) {
            for (int u = 0; u < U; ++u) {
                const std::complex<float>* const in = Ebar_channel.data() + (tbar * U + u) * PD;
                OutT* const out =
                    Ebar_host.data() + (tbar * Fbar_length + std::ptrdiff_t(fi) * U + u) * PD;
                for (std::ptrdiff_t e = 0; e < PD; ++e) {
                    if constexpr (std::is_same_v<OutT, kotekan::int4x2_swapped_withoffset_t>)
                        out[e].val = kotekan::upchan_encode_int4(in[e]);
                    else
                        out[e] = kotekan::upchan_encode_int8(in[e]);
                }
            }
        }
    }

    // Write the result back. Like the real kernel, we fill output channels [0, U*(Fmax-Fmin))
    // and leave any further ones untouched.
    // This copy is synchronous, so it has completed by the time we record the end event, and
    // downstream commands waiting on that event are correctly ordered behind it.
    Ebar_buffer.copy_from_host(Ebar_host, 0, Fbar_length, Tbar_min, Tbar_max);

    return record_end_event();
}

template<typename OutT>
void gpuSimulateCudaUpchannelizerT<OutT>::finalize_frame() {
    E_buffer.finish_read();
    Ebar_buffer.finish_write();
    cudaCommand::finalize_frame();
}

using gpuSimulateCudaUpchannelizer =
    gpuSimulateCudaUpchannelizerT<kotekan::int4x2_swapped_withoffset_t>;
using gpuSimulateCudaUpchannelizer_K8 = gpuSimulateCudaUpchannelizerT<std::complex<std::int8_t>>;

REGISTER_CUDA_COMMAND(gpuSimulateCudaUpchannelizer);
REGISTER_CUDA_COMMAND(gpuSimulateCudaUpchannelizer_K8);
