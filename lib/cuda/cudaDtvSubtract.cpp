#include "cudaDtvSubtract.hpp"

#include "cudaGnssBeamDumpKernel.hpp" // for launch_beam_eig
#include "cudaUtils.hpp"              // for CHECK_CUDA_ERROR
#include "kotekanLogging.hpp"
#include "restServer.hpp"

#include "fmt.hpp"
#include "json.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>

using kotekan::bufferContainer;
using kotekan::Config;
using namespace gnss_cuda;

REGISTER_CUDA_COMMAND(cudaDtvSubtract);

std::shared_ptr<cudaDtvSubtract::Shared> cudaDtvSubtract::shared_for(const std::string& name) {
    static std::mutex mtx;
    static std::map<std::string, std::shared_ptr<Shared>> by_name;
    std::lock_guard<std::mutex> lk(mtx);
    auto& p = by_name[name];
    if (!p)
        p = std::make_shared<Shared>();
    return p;
}

cudaDtvSubtract::cudaDtvSubtract(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, cudaDeviceInterface& device,
                                 int instance_num) :
    cudaCopyToRingbuffer(config, unique_name, host_buffers, device, instance_num,
                         "cudaDtvSubtract"),
    _num_times(config.get<int>(unique_name, "samples_per_data_set")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _num_local_freq(config.get<int>(unique_name, "num_local_freq")),
    _local_channels(config.get<std::vector<std::int32_t>>(unique_name, "dtv_local_channels")),
    _freq_ids(config.get<std::vector<std::int32_t>>(unique_name, "dtv_freq_ids")),
    _out_name(config.get<std::string>(unique_name, "dtv_out_name")),
    _lag_min(config.get_default<int>(unique_name, "dtv_lag_min", -8)),
    _n_lag(config.get_default<int>(unique_name, "dtv_n_lag", 49)),
    _n_src(config.get_default<int>(unique_name, "dtv_sources", 1)),
    _eig_iters(config.get_default<int>(unique_name, "dtv_eig_iters", 24)),
    _layout(dtv_out_layout((int)_local_channels.size(), _n_src, _n_lag)),
    out(_out_name, "dtv_out", std::array<std::ptrdiff_t, 2>{1, (std::ptrdiff_t)_layout.bytes},
        std::array<std::string, 2>{"B", "X"}, std::array<std::ptrdiff_t, 2>{1, 1}, *this),
    _sh(shared_for(unique_name)) {
    const int n = (int)_local_channels.size();
    if (_num_elements != DTV_INPUTS)
        FATAL_ERROR("cudaDtvSubtract: num_elements {:d} != {:d}; the kernels are written for 128 "
                    "inputs",
                    _num_elements, DTV_INPUTS);
    if (_num_times % 256 || _num_times < DTV_HIST)
        FATAL_ERROR("cudaDtvSubtract: samples_per_data_set {:d} must be a multiple of 256",
                    _num_times);
    if ((size_t)_input_size != (size_t)_num_times * _num_local_freq * _num_elements)
        FATAL_ERROR("cudaDtvSubtract: input_size {:d} != samples_per_data_set x num_local_freq x "
                    "num_elements ({:d})",
                    _input_size, (size_t)_num_times * _num_local_freq * _num_elements);
    if (n == 0 || _local_channels.size() != _freq_ids.size())
        FATAL_ERROR("cudaDtvSubtract: dtv_local_channels ({:d}) and dtv_freq_ids ({:d}) must be "
                    "non-empty and the same length",
                    n, (int)_freq_ids.size());
    for (std::int32_t f : _local_channels)
        if (f < 0 || f >= _num_local_freq)
            FATAL_ERROR("cudaDtvSubtract: dtv_local_channels entry {:d} outside [0, {:d}) -- "
                        "LOCAL indices, not freq_ids",
                        f, _num_local_freq);
    if (_n_lag < 1 || _n_lag > DTV_MAX_LAGS || _lag_min < -16
        || _lag_min + _n_lag - 1 > DTV_HIST - 1)
        FATAL_ERROR("cudaDtvSubtract: lags {:d}..{:d} outside [-16, {:d}] or more than {:d}",
                    _lag_min, _lag_min + _n_lag - 1, DTV_HIST - 1, DTV_MAX_LAGS);
    if (_n_src < 1 || _n_src > DTV_MAX_SRC)
        FATAL_ERROR("cudaDtvSubtract: dtv_sources {:d} outside [1, {:d}]", _n_src, DTV_MAX_SRC);
    if (_eig_iters < 1 || _eig_iters > 256)
        FATAL_ERROR("cudaDtvSubtract: dtv_eig_iters {:d} outside [1, 256]", _eig_iters);

    gpu_buffers_used.push_back(std::make_tuple(_out_name, true, false, true));

    // Shared device state, named here so --dry-run sees it; instance 0 initialises it.
    const size_t x_bytes = (size_t)n * _num_times * DTV_INPUTS * sizeof(float2);
    const size_t s_bytes = (size_t)n * _n_src * (DTV_HIST + _num_times) * sizeof(float2);
    const size_t w_bytes = (size_t)n * _n_src * DTV_INPUTS * sizeof(float2);
    const size_t g_bytes = (size_t)n * _n_src * _n_lag * DTV_INPUTS * sizeof(float2);
    const size_t cov_bytes = (size_t)n * DTV_INPUTS * DTV_INPUTS * sizeof(float2);
    const size_t q_bytes = (size_t)n * DTV_INPUTS * sizeof(float2);
    device.get_gpu_memory(unique_name + "_x", x_bytes);
    void* d_s = device.get_gpu_memory(unique_name + "_s", s_bytes);
    void* d_w = device.get_gpu_memory(unique_name + "_w", w_bytes);
    void* d_g = device.get_gpu_memory(unique_name + "_g", g_bytes);
    void* d_act = device.get_gpu_memory(unique_name + "_act", (size_t)n * _n_src * sizeof(int));
    void* d_sel = device.get_gpu_memory(unique_name + "_sel", (size_t)n * sizeof(int));
    device.get_gpu_memory(unique_name + "_cov", cov_bytes);
    device.get_gpu_memory(unique_name + "_q1", q_bytes);
    device.get_gpu_memory(unique_name + "_q2", q_bytes);
    device.get_gpu_memory(unique_name + "_wtmp", q_bytes);
    device.get_gpu_memory(unique_name + "_pwtmp", (size_t)n * DTV_INPUTS * sizeof(float));
    device.get_gpu_memory(unique_name + "_lam", (size_t)n * 4 * sizeof(float));
    device.get_gpu_memory(unique_name + "_ref", (size_t)n * sizeof(int));
    if (instance_num == 0) {
        CHECK_CUDA_ERROR(cudaMemset(d_s, 0, s_bytes));
        CHECK_CUDA_ERROR(cudaMemset(d_w, 0, w_bytes));
        CHECK_CUDA_ERROR(cudaMemset(d_g, 0, g_bytes));
        CHECK_CUDA_ERROR(cudaMemset(d_act, 0, (size_t)n * _n_src * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMemcpy(d_sel, _local_channels.data(), (size_t)n * sizeof(int),
                                    cudaMemcpyHostToDevice));
        std::lock_guard<std::mutex> lk(_sh->mtx);
        _sh->w.assign((size_t)n * _n_src * DTV_INPUTS, make_float2(0.0f, 0.0f));
        _sh->g.assign((size_t)n * _n_src * _n_lag * DTV_INPUTS, make_float2(0.0f, 0.0f));
        _sh->act.assign((size_t)n * _n_src, 0);
        _sh->state_file = config.get_default<std::string>(unique_name, "dtv_state_file", "");
    }

    _hdr_stage.assign(_layout.off_freq + sizeof(std::int32_t) * n, 0);
    auto* hdr = reinterpret_cast<DtvOutHdr*>(_hdr_stage.data());
    hdr->magic = DTV_OUT_MAGIC;
    hdr->version = DTV_OUT_VERSION;
    hdr->n_chan = n;
    hdr->n_times = _num_times;
    hdr->n_inputs = DTV_INPUTS;
    hdr->n_lag = _n_lag;
    hdr->lag_min = _lag_min;
    hdr->n_src = _n_src;
    std::memcpy(_hdr_stage.data() + _layout.off_freq, _freq_ids.data(), sizeof(std::int32_t) * n);
    _lam_stage.assign((size_t)n * 4, 0.0f);
    _w_stage.assign((size_t)n * _n_src * DTV_INPUTS, make_float2(0.0f, 0.0f));
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&_lam_event, cudaEventDisableTiming));

    if (instance_num == 0) {
        std::string err;
        if (!_sh->state_file.empty() && !load_state_file(_sh->state_file, err))
            WARN("cudaDtvSubtract: dtv_state_file {:s}: {:s}; starting without it", _sh->state_file,
                 err);

        auto sh = _sh;
        const int n_src = _n_src;
        auto& rest = kotekan::restServer::instance();
        rest.register_post_callback(
            unique_name + "/start", [this, sh, n_src, unique_name](
                                        kotekan::connectionInstance& conn, nlohmann::json& json) {
                const int k = json.value("source", 0);
                const int frames = json.value("frames", 24);
                if (k < 0 || k >= n_src || frames < 1 || frames > 100000) {
                    conn.send_error("expected {\"source\": 0.." + std::to_string(n_src - 1)
                                        + ", \"frames\": 1..100000}",
                                    kotekan::HTTP_RESPONSE::BAD_REQUEST);
                    return;
                }
                {
                    std::lock_guard<std::mutex> lk(sh->mtx);
                    sh->start_src = k;
                    sh->start_left = sh->start_total = frames;
                }
                INFO("cudaDtvSubtract[{:s}]: eigen start of source {:d} over {:d} frames",
                     unique_name, k, frames);
                nlohmann::json r;
                r["source"] = k;
                r["frames"] = frames;
                conn.send_json_reply(r);
            });
        rest.register_post_callback(
            unique_name + "/load",
            [this, unique_name](kotekan::connectionInstance& conn, nlohmann::json& json) {
                std::string path;
                try {
                    path = json.at("file").get<std::string>();
                } catch (const std::exception& ex) {
                    conn.send_error(std::string("expected {\"file\": <path>}: ") + ex.what(),
                                    kotekan::HTTP_RESPONSE::BAD_REQUEST);
                    return;
                }
                std::string err;
                if (!load_state_file(path, err)) {
                    conn.send_error(err, kotekan::HTTP_RESPONSE::BAD_REQUEST);
                    return;
                }
                INFO("cudaDtvSubtract[{:s}]: loaded {:s}", unique_name, path);
                nlohmann::json r;
                r["file"] = path;
                conn.send_json_reply(r);
            });
        rest.register_post_callback(
            unique_name + "/clear", [this, sh, n_src, n, unique_name](
                                        kotekan::connectionInstance& conn, nlohmann::json& json) {
                const int k = json.value("source", 0);
                if (k < 0 || k >= n_src) {
                    conn.send_error("expected {\"source\": 0.." + std::to_string(n_src - 1) + "}",
                                    kotekan::HTTP_RESPONSE::BAD_REQUEST);
                    return;
                }
                {
                    std::lock_guard<std::mutex> lk(sh->mtx);
                    for (int c = 0; c < n; c++)
                        for (int j = k; j < n_src; j++)
                            sh->act[(size_t)c * n_src + j] = 0;
                    sh->start_src = -1;
                    sh->dirty_w = true;
                }
                INFO("cudaDtvSubtract[{:s}]: sources {:d}.. cleared", unique_name, k);
                nlohmann::json r;
                r["cleared_from"] = k;
                conn.send_json_reply(r);
            });
        rest.register_get_callback(
            unique_name + "/status", [this, sh, n_src, n](kotekan::connectionInstance& conn) {
                nlohmann::json r;
                std::lock_guard<std::mutex> lk(sh->mtx);
                r["frames"] = sh->frames;
                r["state_file"] = sh->state_file;
                r["last_error"] = sh->last_error;
                r["start_source"] = sh->start_src;
                r["start_frames_left"] = sh->start_left;
                r["lag_min"] = _lag_min;
                r["n_lag"] = _n_lag;
                nlohmann::json ch = nlohmann::json::array();
                for (int c = 0; c < n; c++) {
                    nlohmann::json e;
                    e["freq_id"] = _freq_ids[(size_t)c];
                    e["local_channel"] = _local_channels[(size_t)c];
                    e["act"] = std::vector<int>(sh->act.begin() + (size_t)c * n_src,
                                                sh->act.begin() + (size_t)(c + 1) * n_src);
                    ch.push_back(e);
                }
                r["channels"] = ch;
                conn.send_json_reply(r);
            });
    }

    INFO("cudaDtvSubtract: {:d} channels (freq_ids {:d}..{:d}) x {:d} samples, lags {:d}..{:d}, "
         "{:d} source slot(s), output '{:s}' ({:.3f} MB/frame)",
         n, _freq_ids.front(), _freq_ids.back(), _num_times, _lag_min, _lag_min + _n_lag - 1,
         _n_src, _out_name, _layout.bytes / 1.0e6);
}

cudaDtvSubtract::~cudaDtvSubtract() {
    if (_lam_event)
        cudaEventDestroy(_lam_event);
}

bool cudaDtvSubtract::load_state_file(const std::string& path, std::string& err) {
    const int n = (int)_local_channels.size();
    std::ifstream fh(path, std::ios::binary);
    if (!fh) {
        err = "cannot open " + path;
        return false;
    }
    DtvStateHdr h;
    fh.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!fh || h.magic != DTV_STATE_MAGIC || h.version != DTV_STATE_VERSION) {
        err = path + ": not a DTV state file (magic/version)";
        return false;
    }
    if (h.n_inputs != DTV_INPUTS || h.n_lag != _n_lag || h.lag_min != _lag_min || h.n_src < 1
        || h.n_src > DTV_MAX_SRC) {
        err = path + ": lags " + std::to_string(h.lag_min) + "+" + std::to_string(h.n_lag) + " x "
              + std::to_string(h.n_src) + " sources, " + std::to_string(h.n_inputs)
              + " inputs; this command has " + std::to_string(_lag_min) + "+"
              + std::to_string(_n_lag) + " x " + std::to_string(_n_src) + ", 128";
        return false;
    }
    std::vector<float2> w((size_t)n * _n_src * DTV_INPUTS, make_float2(0.0f, 0.0f));
    std::vector<float2> g((size_t)n * _n_src * _n_lag * DTV_INPUTS, make_float2(0.0f, 0.0f));
    std::vector<int> act((size_t)n * _n_src, 0);
    std::vector<float2> rw((size_t)h.n_src * DTV_INPUTS), rg((size_t)h.n_src * _n_lag * DTV_INPUTS);
    int matched = 0;
    for (int r = 0; r < h.n_chan; r++) {
        DtvStateRec rec;
        fh.read(reinterpret_cast<char*>(&rec), sizeof(rec));
        fh.read(reinterpret_cast<char*>(rw.data()), rw.size() * sizeof(float2));
        fh.read(reinterpret_cast<char*>(rg.data()), rg.size() * sizeof(float2));
        if (!fh) {
            err = path + ": truncated at record " + std::to_string(r);
            return false;
        }
        for (int c = 0; c < n; c++) {
            if (_freq_ids[(size_t)c] != rec.freq_id)
                continue;
            matched++;
            const int ks = std::min({rec.n_src_set, h.n_src, _n_src});
            for (int k = 0; k < ks; k++) {
                std::memcpy(&w[((size_t)c * _n_src + k) * DTV_INPUTS], &rw[(size_t)k * DTV_INPUTS],
                            DTV_INPUTS * sizeof(float2));
                std::memcpy(&g[((size_t)c * _n_src + k) * _n_lag * DTV_INPUTS],
                            &rg[(size_t)k * _n_lag * DTV_INPUTS],
                            (size_t)_n_lag * DTV_INPUTS * sizeof(float2));
                act[(size_t)c * _n_src + k] = rec.subtract ? 2 : 1;
            }
        }
    }
    if (matched == 0) {
        err = path + ": none of its " + std::to_string(h.n_chan)
              + " channels is one of this "
                "command's";
        return false;
    }
    std::lock_guard<std::mutex> lk(_sh->mtx);
    _sh->w.swap(w);
    _sh->g.swap(g);
    _sh->act.swap(act);
    _sh->dirty_w = _sh->dirty_g = true;
    _sh->start_src = -1;
    _sh->state_file = path;
    _sh->last_error.clear();
    INFO("cudaDtvSubtract: {:s}: {:d} of this command's {:d} channels loaded", path, matched, n);
    return true;
}

cudaEvent_t cudaDtvSubtract::execute(cudaPipelineState& pipestate,
                                     const std::vector<cudaEvent_t>& pre_events) {
    (void)pipestate;
    pre_execute();
    record_start_event();
    const cudaStream_t stream = device.getStream(cuda_stream_id);
    const int n = (int)_local_channels.size();

    const std::shared_ptr<chordMetadata> meta = copy_frame(pre_events);

    // The frame just enqueued, in the ring. Whole frames: the ring is a multiple of the frame.
    const size_t start = output_cursor % _ring_buffer_size;
    if (start + _input_size > _ring_buffer_size)
        FATAL_ERROR("cudaDtvSubtract: frame at ring offset {:d} wraps the {:d}-byte ring", start,
                    _ring_buffer_size);
    auto* d_frame =
        static_cast<unsigned char*>(device.get_gpu_memory(_gpu_mem_output, _ring_buffer_size))
        + start;
    const long long seq0 = meta ? meta->get_fpga_seq_num() : 0;

    if (!_checked_freqs) {
        if (meta && meta->has_coarse_freq()) {
            const std::vector<int> coarse = meta->get_coarse_freq();
            for (int c = 0; c < n; c++) {
                const int f = _local_channels[(size_t)c];
                if (f >= (int)coarse.size() || coarse[(size_t)f] != _freq_ids[(size_t)c])
                    FATAL_ERROR("cudaDtvSubtract: local channel {:d} carries freq_id {:d}, config "
                                "expects {:d}",
                                f, f < (int)coarse.size() ? coarse[(size_t)f] : -1,
                                _freq_ids[(size_t)c]);
            }
        } else {
            WARN("cudaDtvSubtract: the frames carry no coarse frequencies (test data?); "
                 "dtv_freq_ids are taken on trust");
        }
        _checked_freqs = true;
    }
    if (meta)
        out.set_metadata(meta);

    const size_t x_bytes = (size_t)n * _num_times * DTV_INPUTS * sizeof(float2);
    const size_t s_bytes = (size_t)n * _n_src * (DTV_HIST + _num_times) * sizeof(float2);
    const size_t w_bytes = (size_t)n * _n_src * DTV_INPUTS * sizeof(float2);
    const size_t g_bytes = (size_t)n * _n_src * _n_lag * DTV_INPUTS * sizeof(float2);
    const size_t act_bytes = (size_t)n * _n_src * sizeof(int);
    const size_t cov_bytes = (size_t)n * DTV_INPUTS * DTV_INPUTS * sizeof(float2);
    const size_t q_bytes = (size_t)n * DTV_INPUTS * sizeof(float2);
    auto* d_x = (float2*)device.get_gpu_memory(unique_name + "_x", x_bytes);
    auto* d_s = (float2*)device.get_gpu_memory(unique_name + "_s", s_bytes);
    auto* d_w = (float2*)device.get_gpu_memory(unique_name + "_w", w_bytes);
    auto* d_g = (float2*)device.get_gpu_memory(unique_name + "_g", g_bytes);
    auto* d_act = (int*)device.get_gpu_memory(unique_name + "_act", act_bytes);
    auto* d_sel = (const int*)device.get_gpu_memory(unique_name + "_sel", (size_t)n * sizeof(int));
    auto* d_cov = (float2*)device.get_gpu_memory(unique_name + "_cov", cov_bytes);
    auto* d_q1 = (float2*)device.get_gpu_memory(unique_name + "_q1", q_bytes);
    auto* d_q2 = (float2*)device.get_gpu_memory(unique_name + "_q2", q_bytes);
    auto* d_wtmp = (float2*)device.get_gpu_memory(unique_name + "_wtmp", q_bytes);
    auto* d_pwtmp = (float*)device.get_gpu_memory(unique_name + "_pwtmp",
                                                  (size_t)n * DTV_INPUTS * sizeof(float));
    auto* d_lam =
        (float*)device.get_gpu_memory(unique_name + "_lam", (size_t)n * 4 * sizeof(float));
    auto* d_ref = (int*)device.get_gpu_memory(unique_name + "_ref", (size_t)n * sizeof(int));
    unsigned char* d_out = out.get_ndarray().data();

    // The eigenvalues and weights of a finished eigen start, once their copies have landed.
    if (_lam_pending && cudaEventQuery(_lam_event) == cudaSuccess) {
        _lam_pending = false;
        std::lock_guard<std::mutex> lk(_sh->mtx);
        if (!_sh->dirty_w)
            _sh->w = _w_stage;
        std::string s;
        for (int c = 0; c < n; c++)
            s += fmt::format(" {:d}:{:.1f}/{:.1f}", _freq_ids[(size_t)c], _lam_stage[(size_t)c * 4],
                             _lam_stage[(size_t)c * 4 + 1]);
        INFO("cudaDtvSubtract: eigen start of source {:d} done; whitened lambda1/lambda2 per "
             "freq_id:{:s}",
             _lam_src, s);
    }

    // State changes from the REST side, under the shared lock; the uploads are pageable copies,
    // which return once the bytes are staged.
    int start_src = -1, start_total = 0;
    bool do_cov = false, do_finish = false;
    {
        std::lock_guard<std::mutex> lk(_sh->mtx);
        if (_sh->dirty_w || _sh->dirty_g) {
            if (_sh->dirty_w)
                CHECK_CUDA_ERROR(
                    cudaMemcpyAsync(d_w, _sh->w.data(), w_bytes, cudaMemcpyHostToDevice, stream));
            if (_sh->dirty_g)
                CHECK_CUDA_ERROR(
                    cudaMemcpyAsync(d_g, _sh->g.data(), g_bytes, cudaMemcpyHostToDevice, stream));
            CHECK_CUDA_ERROR(
                cudaMemcpyAsync(d_act, _sh->act.data(), act_bytes, cudaMemcpyHostToDevice, stream));
            _sh->dirty_w = _sh->dirty_g = false;
        }
        if (_sh->start_src >= 0 && _sh->start_left > 0) {
            start_src = _sh->start_src;
            do_cov = true;
            if (_sh->start_left == _sh->start_total) {
                CHECK_CUDA_ERROR(cudaMemsetAsync(d_cov, 0, cov_bytes, stream));
                CHECK_CUDA_ERROR(cudaMemsetAsync(d_q1, 0, q_bytes, stream));
                CHECK_CUDA_ERROR(cudaMemsetAsync(d_q2, 0, q_bytes, stream));
            }
            if (--_sh->start_left == 0) {
                do_finish = true;
                start_total = _sh->start_total;
                _sh->start_src = -1;
                for (int c = 0; c < n; c++)
                    _sh->act[(size_t)c * _n_src + start_src] =
                        std::max(_sh->act[(size_t)c * _n_src + start_src], 1);
                CHECK_CUDA_ERROR(cudaMemcpyAsync(d_act, _sh->act.data(), act_bytes,
                                                 cudaMemcpyHostToDevice, stream));
            }
        }
        _sh->frames++;
    }

    auto* hdr = reinterpret_cast<DtvOutHdr*>(_hdr_stage.data());
    hdr->seq0 = seq0;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_out, _hdr_stage.data(), _hdr_stage.size(),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(
        cudaMemsetAsync(d_out + _layout.off_ps, 0, _layout.bytes - _layout.off_ps, stream));

    launch_dtv_decode(d_frame, d_sel, n, _num_local_freq, _num_times, d_x,
                      (float*)(d_out + _layout.off_pw_raw), stream);
    launch_dtv_shift(d_s, n, _n_src, _num_times, stream);
    for (int k = 0; k < _n_src; k++) {
        if (do_cov && k == start_src) {
            launch_dtv_cov(d_x, n, _num_times, d_cov, stream);
            if (do_finish) {
                launch_dtv_pick_ref(d_cov, n, d_ref, stream);
                launch_beam_eig(d_cov, n, start_total * _num_times, 0, _eig_iters, d_q1, d_q2,
                                d_wtmp, d_pwtmp, d_lam, stream, d_ref);
                CHECK_CUDA_ERROR(cudaMemcpy2DAsync(
                    d_w + (size_t)k * DTV_INPUTS, (size_t)_n_src * DTV_INPUTS * sizeof(float2),
                    d_wtmp, DTV_INPUTS * sizeof(float2), DTV_INPUTS * sizeof(float2), n,
                    cudaMemcpyDeviceToDevice, stream));
                CHECK_CUDA_ERROR(cudaMemcpyAsync(_lam_stage.data(), d_lam,
                                                 _lam_stage.size() * sizeof(float),
                                                 cudaMemcpyDeviceToHost, stream));
                CHECK_CUDA_ERROR(
                    cudaMemcpyAsync(_w_stage.data(), d_w, w_bytes, cudaMemcpyDeviceToHost, stream));
                CHECK_CUDA_ERROR(cudaEventRecord(_lam_event, stream));
                _lam_pending = true;
                _lam_src = k;
            }
        }
        launch_dtv_beam(d_x, d_w, d_act, k, n, _n_src, _num_times, d_s,
                        (float*)(d_out + _layout.off_ps), stream);
        launch_dtv_lagcorr(d_x, d_s, d_act, k, n, _n_src, _num_times, _n_lag, _lag_min,
                           (float2*)(d_out + _layout.off_rxs), (float2*)(d_out + _layout.off_rss),
                           stream);
        launch_dtv_subtract(d_x, d_s, d_g, d_act, k, n, _n_src, _num_times, _n_lag, _lag_min,
                            stream);
    }
    launch_dtv_encode(d_x, d_sel, d_act, n, _n_src, _num_local_freq, _num_times, d_frame,
                      (float*)(d_out + _layout.off_pw_res), stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_out + _layout.off_act, d_act, act_bytes,
                                     cudaMemcpyDeviceToDevice, stream));
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(d_out + _layout.off_w, d_w, w_bytes, cudaMemcpyDeviceToDevice, stream));
    return record_end_event();
}
