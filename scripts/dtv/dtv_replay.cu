// Replay of cudaDtvSubtract's per-frame sequence on recorded 4+4b voltages, with the repo's own
// kernels (lib/cuda/cudaDtvSubtractKernel.cu, cudaGnssBeamDumpKernel.cu): the same uploads, eigen
// start, beam, lag correlations, subtraction (cuBLAS products, TF32 unless --fp32) and
// re-quantisation as cudaDtvSubtract::execute, frame by frame. Writes the stage's output frames
// (rawFileWrite layout with an empty metadata block, as scripts/gnss/dtv_state.py reads them) and
// the residual bytes.
//
//   nvcc -O3 -std=c++17 -o dtv_replay dtv_replay.cu <repo>/lib/cuda/cudaDtvSubtractKernel.cu \
//        <repo>/lib/cuda/cudaGnssBeamDumpKernel.cu -I<repo>/lib/cuda -lcublas
//   dtv_replay --in x.bin --freq 2512,2720,2912 --out out.raw --res res.bin [--state st.bin]
//              [--start K --start-frames M] [--nsrc 2] [--lag-min -8] [--n-lag 49] [--eig-iters 24]
//              [--fp32]
//
// x.bin: uint8 [T][n_chan][128] (T a multiple of 8192), the channels in --freq order.

#include "cudaDtvSubtractKernel.hpp"
#include "cudaGnssBeamDumpKernel.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

using namespace gnss_cuda;

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__,           \
                    __LINE__);                                                                     \
            exit(2);                                                                               \
        }                                                                                          \
    } while (0)

static void die(const std::string& s) {
    fprintf(stderr, "dtv_replay: %s\n", s.c_str());
    exit(1);
}

// load_state_file of cudaDtvSubtract.cpp
static void load_state(const std::string& path, const std::vector<int>& fids, int n_src, int n_lag,
                       int lag_min, std::vector<float2>& w, std::vector<float2>& g,
                       std::vector<int>& act) {
    FILE* fh = fopen(path.c_str(), "rb");
    if (!fh)
        die("cannot open " + path);
    DtvStateHdr h;
    if (fread(&h, sizeof(h), 1, fh) != 1 || h.magic != DTV_STATE_MAGIC
        || h.version != DTV_STATE_VERSION)
        die(path + ": not a DTV state file");
    if (h.n_inputs != DTV_INPUTS || h.n_lag != n_lag || h.lag_min != lag_min)
        die(path + ": lag grid differs");
    const int n = (int)fids.size();
    std::vector<float2> rw((size_t)h.n_src * DTV_INPUTS), rg((size_t)h.n_src * n_lag * DTV_INPUTS);
    int matched = 0;
    for (int r = 0; r < h.n_chan; r++) {
        DtvStateRec rec;
        if (fread(&rec, sizeof(rec), 1, fh) != 1
            || fread(rw.data(), sizeof(float2), rw.size(), fh) != rw.size()
            || fread(rg.data(), sizeof(float2), rg.size(), fh) != rg.size())
            die(path + ": truncated");
        for (int c = 0; c < n; c++) {
            if (fids[c] != rec.freq_id)
                continue;
            matched++;
            const int ks = std::min({rec.n_src_set, h.n_src, n_src});
            for (int k = 0; k < ks; k++) {
                memcpy(&w[((size_t)c * n_src + k) * DTV_INPUTS], &rw[(size_t)k * DTV_INPUTS],
                       DTV_INPUTS * sizeof(float2));
                memcpy(&g[((size_t)c * n_src + k) * n_lag * DTV_INPUTS],
                       &rg[(size_t)k * n_lag * DTV_INPUTS],
                       (size_t)n_lag * DTV_INPUTS * sizeof(float2));
                act[(size_t)c * n_src + k] = rec.subtract ? 2 : 1;
            }
        }
    }
    fclose(fh);
    if (!matched)
        die(path + ": no channel matches");
    fprintf(stderr, "state %s: %d channels loaded\n", path.c_str(), matched);
}

int main(int argc, char** argv) {
    std::string in, out, res, state;
    std::vector<int> fids;
    int start = -1, start_frames = 0, n_src = 2, lag_min = -8, n_lag = 49, eig_iters = 24;
    bool fp32 = false;
    for (int a = 1; a < argc; a++) {
        const std::string k = argv[a];
        auto val = [&]() -> std::string {
            if (a + 1 >= argc)
                die("missing value for " + k);
            return argv[++a];
        };
        if (k == "--in")
            in = val();
        else if (k == "--out")
            out = val();
        else if (k == "--res")
            res = val();
        else if (k == "--state")
            state = val();
        else if (k == "--freq") {
            std::string s = val();
            for (size_t p = 0; p < s.size();) {
                size_t q = s.find(',', p);
                fids.push_back(
                    atoi(s.substr(p, q == std::string::npos ? std::string::npos : q - p).c_str()));
                if (q == std::string::npos)
                    break;
                p = q + 1;
            }
        } else if (k == "--start")
            start = atoi(val().c_str());
        else if (k == "--start-frames")
            start_frames = atoi(val().c_str());
        else if (k == "--nsrc")
            n_src = atoi(val().c_str());
        else if (k == "--lag-min")
            lag_min = atoi(val().c_str());
        else if (k == "--n-lag")
            n_lag = atoi(val().c_str());
        else if (k == "--eig-iters")
            eig_iters = atoi(val().c_str());
        else if (k == "--fp32")
            fp32 = true;
        else
            die("unknown option " + k);
    }
    if (in.empty() || out.empty() || fids.empty())
        die("need --in, --out and --freq");
    const int n = (int)fids.size(), NT = 8192, NI = DTV_INPUTS;
    FILE* fi = fopen(in.c_str(), "rb");
    if (!fi)
        die("cannot open " + in);
    fseek(fi, 0, SEEK_END);
    const long bytes = ftell(fi);
    fseek(fi, 0, SEEK_SET);
    const long frame_bytes = (long)NT * n * NI;
    const int n_frames = (int)(bytes / frame_bytes);
    if (n_frames < 1 || bytes % frame_bytes)
        die(in + ": not a whole number of " + std::to_string(frame_bytes) + "-byte frames");

    const DtvOutLayout L = dtv_out_layout(n, n_src, n_lag);
    std::vector<float2> w((size_t)n * n_src * NI, make_float2(0, 0)),
        g((size_t)n * n_src * n_lag * NI, make_float2(0, 0));
    std::vector<int> act((size_t)n * n_src, 0), sel(n);
    for (int c = 0; c < n; c++)
        sel[c] = c;
    bool dirty = true;
    if (!state.empty())
        load_state(state, fids, n_src, n_lag, lag_min, w, g, act);

    unsigned char *d_frame, *d_out;
    int *d_sel, *d_act, *d_ref;
    float2 *d_x, *d_s, *d_w, *d_g, *d_cov, *d_q1, *d_q2, *d_wtmp;
    float *d_pwtmp, *d_lam;
    CK(cudaMalloc(&d_frame, frame_bytes));
    CK(cudaMalloc(&d_out, L.bytes));
    CK(cudaMalloc(&d_sel, n * sizeof(int)));
    CK(cudaMalloc(&d_act, act.size() * sizeof(int)));
    CK(cudaMalloc(&d_ref, n * sizeof(int)));
    CK(cudaMalloc(&d_x, (size_t)n * NT * NI * sizeof(float2)));
    CK(cudaMalloc(&d_s, (size_t)n * n_src * (DTV_HIST + NT) * sizeof(float2)));
    CK(cudaMalloc(&d_w, w.size() * sizeof(float2)));
    CK(cudaMalloc(&d_g, g.size() * sizeof(float2)));
    CK(cudaMalloc(&d_cov, (size_t)n * NI * NI * sizeof(float2)));
    CK(cudaMalloc(&d_q1, (size_t)n * NI * sizeof(float2)));
    CK(cudaMalloc(&d_q2, (size_t)n * NI * sizeof(float2)));
    CK(cudaMalloc(&d_wtmp, (size_t)n * NI * sizeof(float2)));
    CK(cudaMalloc(&d_pwtmp, (size_t)n * NI * sizeof(float)));
    CK(cudaMalloc(&d_lam, (size_t)n * 4 * sizeof(float)));
    CK(cudaMemset(d_s, 0, (size_t)n * n_src * (DTV_HIST + NT) * sizeof(float2)));
    CK(cudaMemcpy(d_sel, sel.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<unsigned char> frame(frame_bytes), outf(L.bytes), hdr(L.off_freq + 4 * n, 0);
    auto* H = reinterpret_cast<DtvOutHdr*>(hdr.data());
    H->magic = DTV_OUT_MAGIC;
    H->version = DTV_OUT_VERSION;
    H->n_chan = n;
    H->n_times = NT;
    H->n_inputs = NI;
    H->n_lag = n_lag;
    H->lag_min = lag_min;
    H->n_src = n_src;
    memcpy(hdr.data() + L.off_freq, fids.data(), 4 * n);
    FILE* fo = fopen(out.c_str(), "wb");
    FILE* fr = res.empty() ? nullptr : fopen(res.c_str(), "wb");
    if (!fo || (!res.empty() && !fr))
        die("cannot open the outputs");
    const cudaStream_t st = 0;
    int left = start >= 0 ? start_frames : 0;
    std::vector<int> lags;
    for (int c = 0; c < n; c++) {
        lags.push_back(lag_min);
        lags.push_back(n_lag);
    }
    int* d_lags;
    float2* d_S;
    CK(cudaMalloc(&d_lags, lags.size() * sizeof(int)));
    CK(cudaMalloc(&d_S, (size_t)n * n_lag * NT * sizeof(float2)));
    CK(cudaMemcpy(d_lags, lags.data(), lags.size() * sizeof(int), cudaMemcpyHostToDevice));
    cublasHandle_t blas;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS
        || cublasSetMathMode(blas, fp32 ? CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH)
               != CUBLAS_STATUS_SUCCESS)
        die("cannot set up cuBLAS");
    std::vector<float2> g_up(g.size());
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    float ms_total = 0.0f;

    for (int f = 0; f < n_frames; f++) {
        if (fread(frame.data(), 1, frame_bytes, fi) != (size_t)frame_bytes)
            die("short read");
        CK(cudaMemcpy(d_frame, frame.data(), frame_bytes, cudaMemcpyHostToDevice));
        CK(cudaEventRecord(t0, st));
        if (dirty) {
            CK(cudaMemcpyAsync(d_w, w.data(), w.size() * sizeof(float2), cudaMemcpyHostToDevice,
                               st));
            // as the stage does: g zero where its source does not subtract
            g_up = g;
            for (size_t ck = 0; ck < act.size(); ck++)
                if (act[ck] < 2)
                    std::fill(g_up.begin() + ck * n_lag * NI, g_up.begin() + (ck + 1) * n_lag * NI,
                              make_float2(0, 0));
            CK(cudaMemcpyAsync(d_g, g_up.data(), g_up.size() * sizeof(float2),
                               cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(d_act, act.data(), act.size() * sizeof(int), cudaMemcpyHostToDevice,
                               st));
            dirty = false;
        }
        bool do_cov = false, do_finish = false;
        if (start >= 0 && left > 0) {
            do_cov = true;
            if (left == start_frames) {
                CK(cudaMemsetAsync(d_cov, 0, (size_t)n * NI * NI * sizeof(float2), st));
                CK(cudaMemsetAsync(d_q1, 0, (size_t)n * NI * sizeof(float2), st));
                CK(cudaMemsetAsync(d_q2, 0, (size_t)n * NI * sizeof(float2), st));
            }
            if (--left == 0) {
                do_finish = true;
                for (int c = 0; c < n; c++)
                    act[(size_t)c * n_src + start] = std::max(act[(size_t)c * n_src + start], 1);
                CK(cudaMemcpyAsync(d_act, act.data(), act.size() * sizeof(int),
                                   cudaMemcpyHostToDevice, st));
            }
        }
        H->seq0 = (int64_t)f * NT;
        CK(cudaMemcpyAsync(d_out, hdr.data(), hdr.size(), cudaMemcpyHostToDevice, st));
        CK(cudaMemsetAsync(d_out + L.off_ps, 0, L.bytes - L.off_ps, st));
        launch_dtv_decode(d_frame, d_sel, n, n, NT, d_x, (float*)(d_out + L.off_pw_raw), st);
        launch_dtv_shift(d_s, n, n_src, NT, st);
        for (int k = 0; k < n_src; k++) {
            if (do_cov && k == start) {
                launch_dtv_cov(d_x, n, NT, d_cov, st);
                if (do_finish) {
                    launch_dtv_pick_ref(d_cov, n, d_ref, st);
                    launch_beam_eig(d_cov, n, start_frames * NT, 0, eig_iters, d_q1, d_q2, d_wtmp,
                                    d_pwtmp, d_lam, st, d_ref);
                    CK(cudaMemcpy2DAsync(d_w + (size_t)k * NI, (size_t)n_src * NI * sizeof(float2),
                                         d_wtmp, NI * sizeof(float2), NI * sizeof(float2), n,
                                         cudaMemcpyDeviceToDevice, st));
                }
            }
            launch_dtv_beam(d_x, d_w, d_act, k, n, n_src, NT, d_s, (float*)(d_out + L.off_ps), st);
            launch_dtv_lagged(d_s, d_act, d_lags, k, n, n_src, NT, n_lag, d_S,
                              (float2*)(d_out + L.off_rss), st);
            if (dtv_lagcorr(blas, d_x, d_S, k, n, n_src, NT, n_lag, (float2*)(d_out + L.off_rxs))
                    != CUBLAS_STATUS_SUCCESS
                || dtv_subtract(blas, d_x, d_S, d_g, k, n, n_src, NT, n_lag)
                       != CUBLAS_STATUS_SUCCESS)
                die("cuBLAS product failed");
        }
        launch_dtv_encode(d_x, d_sel, d_act, n, n_src, n, NT, d_frame,
                          (float*)(d_out + L.off_pw_res), st);
        CK(cudaMemcpyAsync(d_out + L.off_act, d_act, act.size() * sizeof(int),
                           cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(d_out + L.off_w, d_w, w.size() * sizeof(float2),
                           cudaMemcpyDeviceToDevice, st));
        CK(cudaEventRecord(t1, st));
        CK(cudaStreamSynchronize(st));
        float ms;
        CK(cudaEventElapsedTime(&ms, t0, t1));
        ms_total += ms;
        if (do_finish) {
            std::vector<float> lam((size_t)n * 4);
            std::vector<int> ref(n);
            CK(cudaMemcpy(lam.data(), d_lam, lam.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ref.data(), d_ref, n * sizeof(int), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(w.data(), d_w, w.size() * sizeof(float2), cudaMemcpyDeviceToHost));
            for (int c = 0; c < n; c++)
                fprintf(stderr,
                        "frame %d: eigen start of source %d, freq_id %d: reference input %d, "
                        "whitened lambda1/lambda2 %.1f/%.1f\n",
                        f, start, fids[c], ref[c], lam[c * 4], lam[c * 4 + 1]);
        }
        CK(cudaMemcpy(outf.data(), d_out, L.bytes, cudaMemcpyDeviceToHost));
        const uint32_t msize = 0;
        fwrite(&msize, 4, 1, fo);
        fwrite(outf.data(), 1, L.bytes, fo);
        if (fr) {
            CK(cudaMemcpy(frame.data(), d_frame, frame_bytes, cudaMemcpyDeviceToHost));
            fwrite(frame.data(), 1, frame_bytes, fr);
        }
    }
    fclose(fo);
    if (fr)
        fclose(fr);
    fclose(fi);
    fprintf(stderr, "%d frames x %d channels x %d source slots: GPU time %.3f ms per frame\n",
            n_frames, n, n_src, ms_total / n_frames);
    return 0;
}
