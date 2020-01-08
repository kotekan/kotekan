#include "hccGPUThread.hpp"

#include "errors.h"
#include "util.h" // for e_time

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <random>
#include <vector>

#ifdef WITH_HCC
#include <hcc/hc.hpp>

#define n_timesteps 65536 // 256*1024
#define n_integrate 32768
#define n_presum 1024 // 16384
#define n_elem 2048
#define n_blk 2080 //(n_elem/32)*(n_elem/32+1)/2

using kotekan::Config;
using kotekan::Stage;

// clang-format off

hccGPUThread::hccGPUThread(Config& config_, Buffer& in_buf_, Buffer& out_buf_, uint32_t gpu_id_) :
    Stage(config_, std::bind(&hccGPUThread::main_thread, this)),
    in_buf(in_buf_), out_buf(out_buf_), gpu_id(gpu_id_) {

}

hccGPUThread::~hccGPUThread() {
    free(_zeros);
}

void hccGPUThread::main_thread() {

    //find accelerators
    std::vector<hc::accelerator> all_accelerators = hc::accelerator::get_all();
    std::vector<hc::accelerator> accelerators;
    std::vector<hc::accelerator_view> acc_views;
    for (auto a = all_accelerators.begin(); a != all_accelerators.end(); a++) {
        // only pick accelerators supported by the HSA runtime
        if (a->is_hsa_accelerator()) {
          INFO("hccGPUThread: Found a GPU!");
          accelerators.push_back(*a);
          acc_views.push_back(a->create_view());
        }
    }
    INFO("hccGPUThread: Got {:d} views.", (int)acc_views.size());

    // Create the index map
    _zeros = (int*)calloc(32*32*n_blk*2,4);
    unsigned int blk_map[n_blk][2];
    int blkid=0;
    for (int y=0; blkid<n_blk; y++) {
        for (int x=y; x<n_elem/32; x++) {
            blk_map[blkid][0] = x;
            blk_map[blkid][1] = y;
            blkid++;
        }
    }

    //host-side arrays
    std::vector<int> corr(32*32*n_blk*2);
    //gpu-side arrays
    hc::array<uint, 2> a_input(hc::extent<2>(n_timesteps,n_elem/4), acc_views.at(gpu_id));
    hc::array<int> a_corr((n_blk*32*32*2), acc_views.at(gpu_id));
    hc::array<int,2> a_presum(n_elem,2, acc_views.at(gpu_id));
    hc::array<uint,2> a_map(hc::extent<2>(n_blk,2), acc_views.at(gpu_id));

    hc::completion_future copy_map = hc::copy_async((int*)blk_map, (int*)blk_map+n_blk*2, a_map);
    copy_map.wait();

      //tiling / workgroup
    hc::extent<3> global_extent(n_blk,4*n_timesteps/n_integrate,16);
    hc::tiled_extent<3> t_extent = global_extent.tile(1,4,16);

    //pre-sum
    hc::extent<2> preglobal_extent(n_timesteps/n_presum,n_elem/4);
    hc::tiled_extent<2> pret_extent = preglobal_extent.tile(1,64);

    int in_buffer_id = 0;
    int out_buffer_id = 0;

    while (!stop_thread) {

        in_buffer_id = get_full_buffer_from_list(&in_buf, &in_buffer_id, 1);
        // If buffer id is -1, then all the producers are done.
        if (in_buffer_id == -1) {
            break;
        }

        INFO("hccGPUThread: gpu {:d}; got full buffer ID {:d}", gpu_id, in_buffer_id);
        double start_time = e_time();

        int * in = (int *)in_buf.frames[in_buffer_id];

        // transfer things in
        hc::completion_future copy_input = hc::copy_async((int*)in, (int*)in+n_elem*n_timesteps/4, a_input);
        copy_input.wait();
        hc::completion_future empty_out = hc::copy_async(_zeros, _zeros + n_elem*2, a_presum);
        empty_out.wait();

        // ----------------- Kernels ----------------
        hc::completion_future kernel_future;
        // Presum data kernel
        kernel_future = hc::parallel_for_each(acc_views.at(gpu_id), pret_extent, [&](hc::tiled_index<2> idx) [[hc]] {
            uint presum_re[4]={0,0,0,0};
            uint presum_im[4]={0,0,0,0};

            for (uint t=0; t<n_presum; t++){
                uint xv=a_input(t+idx.global[0]*n_presum,idx.global[1]);
                #pragma unroll
                for (uint e=0; e<4; e++){
                    presum_im[e]+=hc::__bitextract_u32(xv,e*8+0,4);
                    presum_re[e]+=hc::__bitextract_u32(xv,e*8+4,4);
                }
            }
            for (uint e=0; e<4; e++){
                hc::atomic_fetch_add(a_presum.data()+(idx.global[1]*4+e)*2+0,presum_re[e]);
                hc::atomic_fetch_add(a_presum.data()+(idx.global[1]*4+e)*2+1,presum_im[e]);
            }
         });
         kernel_future.wait();

         // N^2 kernel
         kernel_future = hc::parallel_for_each(acc_views.at(gpu_id), t_extent, [&](hc::tiled_index<3> idx) [[hc]] {
            //figure out where to load data from
            int ix = idx.local[2]/2;
            int iy = idx.local[1]*2 + (idx.local[2]&0x1);
            int input_x = a_map(idx.global[0],0)*8 + ix;
            int input_y = a_map(idx.global[0],1)*8 + iy;

            if (idx.tile[1] == 0) {
                for (int y=0; y<4; y++) for (int x=0; x<4; x++){
                    a_corr(((idx.global[0]*1024 + (iy*4+y)*32 + ix*4+x)*2)+0) =
                          128 * n_timesteps - 8*(a_presum(input_x*4+x,0)+a_presum(input_y*4+y,0) +
                                                 a_presum(input_x*4+x,1)+a_presum(input_y*4+y,1));
                    a_corr(((idx.global[0]*1024 + (iy*4+y)*32 + ix*4+x)*2)+1) =
                                              8*(a_presum(input_x*4+x,0)-a_presum(input_y*4+y,0) -
                                                 a_presum(input_x*4+x,1)+a_presum(input_y*4+y,1));
                }
            }

            //seed the 8x8 workgroup with staggered time offsets
            uint T = ((ix + iy) % 8) + idx.tile[1]*n_integrate;

            //find the address of the work items to hand off to
            //there's gotta be a better way to get this...
            uint dest_x = (((iy-1)&0x6)<<3) + (idx.local[2]^0x1);

            //temporary registers that hold the inputs; y is packed x is not
            uint y_ri[4], x_ir[4], x_ii[2], y_0r[4];

            tile_static uint locoflow_r[4][16][4][2];
            tile_static uint locoflow_i[4][16][4];
            for (; T<n_integrate + idx.tile[1]*n_integrate; T+=16384){
                //zero the accumulation buffers
                uint corr_rr_ri[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
                uint corr_1i_2i[4][2] = {{0,0},    {0,0},    {0,0},    {0,0}    };

                for (int k=0; k<4; k++)
                    locoflow_i[idx.local[1]][idx.local[2]][k]    =
                    locoflow_r[idx.local[1]][idx.local[2]][k][0] =
                    locoflow_r[idx.local[1]][idx.local[2]][k][1] = 0;

                //big 'ol outer loop, to do arbitrarily long accumulations
                for (uint t=0; t<16384; t+=64){
                    //move top bits to overflow to make room in accumulation buffers
                    //had to move to before the loop; after, it jumbled the compilation
                    for (int y=0; y<4; y++)
                        hc::atomic_fetch_add(&locoflow_i[idx.local[1]][idx.local[2]][y],
                                          ((corr_1i_2i[y][0] & 0x80008000) >> 15) +
                                          ((corr_1i_2i[y][1] & 0x80008000) >>  7));

                    for (int y=0; y<4; y++) for (int x=0; x<2; x++)
                        corr_1i_2i[y][x]=corr_1i_2i[y][x] & 0x7fff7fff;

                    for (int y=0; y<4; y++)
                        hc::atomic_fetch_add(&locoflow_r[idx.local[1]][idx.local[2]][y][0],
                                      ((corr_rr_ri[y][0] & 0x80008000) >> 15) +
                                      ((corr_rr_ri[y][1] & 0x80008000) >>  7));
                    for (int y=0; y<4; y++)
                        hc::atomic_fetch_add(&locoflow_r[idx.local[1]][idx.local[2]][y][1],
                                      ((corr_rr_ri[y][2] & 0x80008000) >> 15) +
                                      ((corr_rr_ri[y][3] & 0x80008000) >>  7));
                    for (int y=0; y<4; y++) for (int x=0; x<4; x++)
                        corr_rr_ri[y][x]=corr_rr_ri[y][x] & 0x7fff7fff;

                    //accumulate 256 samples before unpacking
                    #pragma nounroll
                    for (int j=0; j<64; j+=8){
                        //load up 4 inputs from each side of the 4x4 block
                        uint xv = a_input(T+t+j,input_x);//[idx.local[1]][idx.local[2]]);
                        //x_ir == [000I000R]
                        x_ir[0] = hc::__mad24(0x10000u,hc::__bitextract_u32(xv, 0,4),
                                                       hc::__bitextract_u32(xv, 4,4));
                        x_ir[1] = hc::__mad24(0x10000u,hc::__bitextract_u32(xv, 8,4),
                                                       hc::__bitextract_u32(xv,12,4));
                        x_ir[2] = hc::__mad24(0x10000u,hc::__bitextract_u32(xv,16,4),
                                                       hc::__bitextract_u32(xv,20,4));
                        x_ir[3] = hc::__mad24(0x10000u,hc::__bitextract_u32(xv,24,4),
                                                       hc::__bitextract_u32(xv,28,4));
                        //x_ii == [000I000I]
                        x_ii[0] = hc::__mad24(0x10000u,hc::__bitextract_u32(xv, 8,4),
                                                       hc::__bitextract_u32(xv, 0,4));
                        x_ii[1] = hc::__mad24(0x10000u,hc::__bitextract_u32(xv,24,4),
                                                       hc::__bitextract_u32(xv,16,4));

                        uint yv = a_input(T+t+j,input_y);//[idx.local[1]][idx.local[2]]);
                        //y_ri == [000R000I]
                        y_ri[0] = hc::__mad24(0x10000u,hc::__bitextract_u32(yv, 4,4),
                                                       hc::__bitextract_u32(yv, 0,4));
                        y_ri[1] = hc::__mad24(0x10000u,hc::__bitextract_u32(yv,12,4),
                                                       hc::__bitextract_u32(yv, 8,4));
                        y_ri[2] = hc::__mad24(0x10000u,hc::__bitextract_u32(yv,20,4),
                                                       hc::__bitextract_u32(yv,16,4));
                        y_ri[3] = hc::__mad24(0x10000u,hc::__bitextract_u32(yv,28,4),
                                                       hc::__bitextract_u32(yv,24,4));

                        //process 8 timesteps before reloading
                        #pragma unroll //comment this out if you want to read / debug the dump.isa...
                        for (int i=0; i<8; i++){
                            //y_0r == [0000000R] -- same price to re-calculate, and it saves registers!
                            for (int k=0; k<4; k++)
                                y_0r[k] = y_ri[k]>>16;

                            //multiplies!
                            for (int y=0; y<4; y++) for (int x=0; x<4; x++)
                                corr_rr_ri[y][x] = hc::__mad24(x_ir[x],y_ri[y],corr_rr_ri[y][x]);
                            for (int y=0; y<4; y++) for (int x=0; x<2; x++)
                                corr_1i_2i[y][x] = hc::__mad24(x_ii[x],y_0r[y],corr_1i_2i[y][x]);

                            //then pass data
                            for (int k=0; k<4; k++)
                                x_ir[k] = hc::__shfl(x_ir[k],dest_x);
                            for (int k=0; k<4; k++)
                                y_ri[k] = hc::__amdgcn_move_dpp(y_ri[k],0x122,0xf,0xf,0); //rotate right by 2

                            x_ii[0] = hc::__shfl(x_ii[0],dest_x);
                            x_ii[1] = hc::__shfl(x_ii[1],dest_x);
                        }
                    }
                }
                //unpacked into long-term real, imaginary accumulation buffer.
                //this loop eats a ton of VGPRs, use offsets to prevent pre-allocation
                int *out=a_corr.data() + (idx.global[0]*1024 + iy*32*4 + ix*4)*2;
                out+=y_ri[0]+73; //stopping pre-VGRP allocation
                #pragma unroll
                for (int y=0; y<4; y++){
                  uint r[4] = {
                      (corr_rr_ri[y][0]>>16) +
                        (hc::__bitextract_u32(locoflow_r[idx.local[1]][idx.local[2]][y][0],16,8) << 15),
                      (corr_rr_ri[y][1]>>16) +
                        (hc::__bitextract_u32(locoflow_r[idx.local[1]][idx.local[2]][y][0],24,8) << 15),
                      (corr_rr_ri[y][2]>>16) +
                        (hc::__bitextract_u32(locoflow_r[idx.local[1]][idx.local[2]][y][1],16,8) << 15),
                      (corr_rr_ri[y][3]>>16) +
                        (hc::__bitextract_u32(locoflow_r[idx.local[1]][idx.local[2]][y][1],24,8) << 15)
                  };
                  uint i[4] = {
                      (hc::__bitextract_u32(locoflow_i[idx.local[1]][idx.local[2]][y],    0,8) << 15) -
                      (hc::__bitextract_u32(locoflow_r[idx.local[1]][idx.local[2]][y][0], 0,8) << 15) +
                       hc::__bitextract_u32(corr_1i_2i[y][0], 0,16) - (corr_rr_ri[y][0]&0xffff),
                      (hc::__bitextract_u32(locoflow_i[idx.local[1]][idx.local[2]][y],   16,8) << 15) -
                      (hc::__bitextract_u32(locoflow_r[idx.local[1]][idx.local[2]][y][0], 8,8) << 15) +
                       hc::__bitextract_u32(corr_1i_2i[y][0],16,16) - (corr_rr_ri[y][1]&0xffff),
                      (hc::__bitextract_u32(locoflow_i[idx.local[1]][idx.local[2]][y],    8,8) << 15) -
                      (hc::__bitextract_u32(locoflow_r[idx.local[1]][idx.local[2]][y][1], 0,8) << 15) +
                       hc::__bitextract_u32(corr_1i_2i[y][1], 0,16) - (corr_rr_ri[y][2]&0xffff),
                      (hc::__bitextract_u32(locoflow_i[idx.local[1]][idx.local[2]][y],   24,8) << 15) -
                      (hc::__bitextract_u32(locoflow_r[idx.local[1]][idx.local[2]][y][1], 8,8) << 15) +
                       hc::__bitextract_u32(corr_1i_2i[y][1],16,16) - (corr_rr_ri[y][3]&0xffff)
                  };
                  #pragma unroll
                  for (int x=0; x<4; x++){
                      hc::atomic_fetch_add(out++ -y_ri[0]-73,r[x]); //stopping pre-VGRP allocation
                      hc::atomic_fetch_add(out++ -y_ri[0]-73,i[x]); //stopping pre-VGRP allocation
                      // hc::atomic_fetch_add(out++,r[x]);
                      // hc::atomic_fetch_add(out++,i[x]);
                  }
                  out+=56; //(32-4)*2;
                }
            }
        });
        kernel_future.wait();

        // ----------------- End Kernels ----------------

        INFO("hccGPUTHread: gpu {:d}; Wait for empty output buffer id {:d}", gpu_id ,out_buffer_id);
        wait_for_empty_frame(&out_buf, unique_name, out_buffer_id);

        // copy the data on the GPU back to the host
        int * out = (int*)out_buf.frames[out_buffer_id];
        hc::completion_future corr_f = hc::copy_async(a_corr, out);
        corr_f.wait();

        double end_time = e_time();
        INFO("hccGPUThread: gpu {:d}; Finished GPU exec for buffer id {:d}, time {:f}s, expected time {:f}s", gpu_id, in_buffer_id, end_time-start_time, 0.00000256 * (double)65536);


        INFO("hccGPUThread: gpu {:d}; copied data back with buffer id {:d}", gpu_id ,in_buffer_id);

        // Copy the information contained in the input buffer
        move_buffer_info(&in_buf, in_buffer_id,
                         &out_buf, out_buffer_id);

        // Mark the input buffer as "empty" so that it can be reused.
        mark_frame_empty(&in_buf, unique_name.c_str(), in_buffer_id);

        // Mark the output buffer as full, so it can be processed.
        mark_frame_full(&out_buf, unique_name.c_str(), out_buffer_id);

        in_buffer_id = (in_buffer_id + 1) % in_buf.num_frames;
        out_buffer_id = (out_buffer_id + 1) % out_buf.num_frames;
    }

    INFO("hccGPUThread: exited main thread");
}

#else  // For building on systems without HCC.

hccGPUThread::hccGPUThread(Config& config_, Buffer& in_buf_, Buffer& out_buf_, uint32_t gpu_id_):
    Stage(config_, std::bind(&hccGPUThread::main_thread, this)),
    in_buf(in_buf_), out_buf(out_buf_), gpu_id(gpu_id_) {

    ERROR("HCC wasn't built.");
}

void hccGPUThread::main_thread() {
    ERROR("HCC wasn't built.")
}

hccGPUThread::~hccGPUThread() {
}

// clang-format on

#endif
