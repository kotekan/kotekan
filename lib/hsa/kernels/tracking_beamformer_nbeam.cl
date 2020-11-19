#define TS 64
#define RE x
#define IM y

__kernel void trackingbf_float( __global uint *data,
                               __global float2 *phase,
                               __global uchar *output,
                               __global float *scaling){
    float2 sum;
    uint nsamp = get_global_size(2)*TS;

    float2 ph[4][4];//[4-in-a-word][4-words-per-work-item]

    uint pol = get_group_id(0);
    uint beam_id = get_global_id(1);
    for (int b=get_global_id(1); b<(get_global_id(1)+1); b++){
        for (int tt = 0; tt<4; tt++){
            uint element = get_local_id(0)*4 + tt;
            ph[0][tt] = phase[ (b*2+pol)*1024 + element*4];
            ph[1][tt] = phase[ (b*2+pol)*1024 + element*4+1];
            ph[2][tt] = phase[ (b*2+pol)*1024 + element*4+2];
            ph[3][tt] = phase[ (b*2+pol)*1024 + element*4+3];
        }
        for (int t=0; t<TS; t++){
            sum.RE=0.0;
            sum.IM=0.0;
            for (int tt = 0; tt<4; tt++){

                uint element = get_local_id(0)*4 + tt;
                uint data_temp = data[(t*get_global_size(2)+get_group_id(2))*512 + pol*256 + element];

                sum.RE +=
                      ph[0][tt].RE*((float)((data_temp & 0x000000f0)>> 4u)-8)
                    + ph[0][tt].IM*((float)((data_temp & 0x0000000f)>> 0u)-8)
                    + ph[1][tt].RE*((float)((data_temp & 0x0000f000)>>12u)-8)
                    + ph[1][tt].IM*((float)((data_temp & 0x00000f00)>> 8u)-8)
                    + ph[2][tt].RE*((float)((data_temp & 0x00f00000)>>20u)-8)
                    + ph[2][tt].IM*((float)((data_temp & 0x000f0000)>>16u)-8)
                    + ph[3][tt].RE*((float)((data_temp & 0xf0000000)>>28u)-8)
                    + ph[3][tt].IM*((float)((data_temp & 0x0f000000)>>24u)-8);

                sum.IM +=
                    - ph[0][tt].IM*((float)((data_temp & 0x000000f0)>> 4u)-8)
                    + ph[0][tt].RE*((float)((data_temp & 0x0000000f)>> 0u)-8)
                    - ph[1][tt].IM*((float)((data_temp & 0x0000f000)>>12u)-8)
                    + ph[1][tt].RE*((float)((data_temp & 0x00000f00)>> 8u)-8)
                    - ph[2][tt].IM*((float)((data_temp & 0x00f00000)>>20u)-8)
                    + ph[2][tt].RE*((float)((data_temp & 0x000f0000)>>16u)-8)
                    - ph[3][tt].IM*((float)((data_temp & 0xf0000000)>>28u)-8)
                    + ph[3][tt].RE*((float)((data_temp & 0x0f000000)>>24u)-8);
            }

            //Reduction of 64, eventually each number comes from the sum of 1024 values
            sum.RE += as_float(__builtin_amdgcn_ds_bpermute((16+get_local_id(0))*4,as_uint(sum.RE)))+
                      as_float(__builtin_amdgcn_ds_bpermute((32+get_local_id(0))*4,as_uint(sum.RE)))+
                      as_float(__builtin_amdgcn_ds_bpermute((48+get_local_id(0))*4,as_uint(sum.RE)));
            sum.IM += as_float(__builtin_amdgcn_ds_bpermute((16+get_local_id(0))*4,as_uint(sum.IM)))+
                      as_float(__builtin_amdgcn_ds_bpermute((32+get_local_id(0))*4,as_uint(sum.IM)))+
                      as_float(__builtin_amdgcn_ds_bpermute((48+get_local_id(0))*4,as_uint(sum.IM)));

            //Adding 4,8,12
            sum.RE += as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE),0x104,0xf,0xf,0))+
                      as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE),0x108,0xf,0xf,0))+
                      as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE),0x10c,0xf,0xf,0));
            sum.IM += as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM),0x104,0xf,0xf,0))+
                      as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM),0x108,0xf,0xf,0))+
                      as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM),0x10c,0xf,0xf,0));

            //Adding 1,2,3
            sum.RE += as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE),0x101,0xf,0xf,0))+
                      as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE),0x102,0xf,0xf,0))+
                      as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.RE),0x103,0xf,0xf,0));
            sum.IM += as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM),0x101,0xf,0xf,0))+
                      as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM),0x102,0xf,0xf,0))+
                      as_float(__builtin_amdgcn_mov_dpp(as_uint(sum.IM),0x103,0xf,0xf,0));

            if (get_local_id(0) == 0) {
                // Scale and offset encode the values
                sum.RE = sum.RE / (scaling[beam_id]) + 8;
                sum.IM = sum.IM / (scaling[beam_id]) + 8;

                // Clamp the values to [0,15]
                sum.RE = (sum.RE > 15) ? 15 : sum.RE;
                sum.RE = (sum.RE < 0) ? 0 : sum.RE;

                sum.IM = (sum.IM > 15) ? 15 : sum.IM;
                sum.IM = (sum.IM < 0) ? 0 : sum.IM;

                output[(t * nsamp / TS + get_group_id(2)) * get_global_size(1) * 2 + b * 2
                       + get_group_id(0)] = (((int)sum.RE << 4) & 0xF0)
                                            + ((int)sum.IM & 0x0F);
            }
        } //end t
    }
}

