//#define SAMPLES_PER_DATA_SET
//#define NUM_ELEMENTS
//#define NUM_FREQS
//#define BLOCK_SIZE
//#define COARSE_BLOCK_SIZE
//#define WI_SIZE



#define SDOT8


#if COARSE_BLOCK_SIZE > 8
#error "__builtin_amdgcn_ds_bpermute won't work with this workgroup size!"
#endif

#define xl get_local_id(0)
#define yl get_local_id(1)
#define zl get_local_id(2)

#define xg get_global_id(0)
#define yg get_global_id(1)
#define zg get_global_id(2)

#define xgr get_group_id(0)
#define ygr get_group_id(1)
#define zgr get_group_id(2)

#define NUM_FREQS get_num_groups(2)
//get_num_groups(1)
#define NUM_BLOCKS get_num_groups(1)

#define FREQ_ID zgr
#define BLOCK_ID ygr

//the input is [Freqs, Time/8, Input, 8-times, 2-re-im]

int dot4b(uint,uint);
int dot4b(uint a, uint b){
    int sum=0;
    for (int i=0; i<8; i++){
        int a_s = ((a>>(4*i)) & 0x7);
        a_s    -= ((a>>(4*i)) & 0x8);
        int b_s = ((b>>(4*i)) & 0x7);
        b_s    -= ((b>>(4*i)) & 0x8);
        sum += a_s * b_s;
    }
    return sum;
}

__kernel __attribute__((reqd_work_group_size(COARSE_BLOCK_SIZE, COARSE_BLOCK_SIZE, 1)))
void corr ( __global const uint *input,
            __global int *presum,
            __global int *corr_buf,
            __global const uint *id_x_map,
            __global const uint *id_y_map)
{
    //figure out where to load data from, 1st x,y elem to load
    uint addr_x = (id_x_map[BLOCK_ID]*COARSE_BLOCK_SIZE + xl)*WI_SIZE;
    uint addr_y = (id_y_map[BLOCK_ID]*COARSE_BLOCK_SIZE + yl)*WI_SIZE;

    //seed the workgroup with staggered time offsets
    uint t = ((xl + yl) % COARSE_BLOCK_SIZE);

    //find the address of the work items to hand off to
    uint dest_x = ((yl+1)%COARSE_BLOCK_SIZE)*COARSE_BLOCK_SIZE + xl;
    uint dest_y = ((xl+1)%COARSE_BLOCK_SIZE) + yl*COARSE_BLOCK_SIZE;

    //temporary registers that hold the inputs; y is packed x is not
    uint xr[WI_SIZE], xi[WI_SIZE], yr[WI_SIZE], yi[WI_SIZE];

    //zero the accumulation buffers
    int corr_r[WI_SIZE][WI_SIZE];
    int corr_i[WI_SIZE][WI_SIZE];

    #pragma unroll
    for (int i=0; i<WI_SIZE; i++)
        for (int j=0; j<WI_SIZE; j++) {
            corr_r[i][j]=0;
            corr_i[i][j]=0;
        }

    //accumulate
    for (int j=0; j<SAMPLES_PER_DATA_SET/8; j+=COARSE_BLOCK_SIZE){
        //input is [Freqs, Time/8, Input, 8-times, 2-re-im]
        for (int i=0; i<WI_SIZE; i++) {
            xr[i] = input[(((j + t)*NUM_FREQS 
                               + FREQ_ID) * NUM_ELEMENTS 
                               + addr_x + i) * 2];
            xi[i] = input[(((j + t)*NUM_FREQS 
                               + FREQ_ID) * NUM_ELEMENTS 
                               + addr_x + i) * 2 + 1];
            yr[i] = input[(((j + t)*NUM_FREQS 
                               + FREQ_ID) * NUM_ELEMENTS 
                               + addr_y + i) * 2];
            yi[i] = input[(((j + t)*NUM_FREQS 
                               + FREQ_ID) * NUM_ELEMENTS 
                               + addr_y + i) * 2 + 1];
        }

        //process COARSE_BLOCK_SIZE timesteps before reloading
        for (int i=0; i<COARSE_BLOCK_SIZE; i++){
            #pragma unroll
            for (int y=0; y<WI_SIZE; y++) for (int x=0; x<WI_SIZE; x++) {
#ifndef SDOT8
                corr_r[y][x] += dot4b(xr[x],yr[y]);
                corr_r[y][x] += dot4b(xi[x],yi[y]);
                corr_i[y][x] += dot4b(xr[x],yi[y]);
                corr_i[y][x] -= dot4b(xi[x],yr[y]);
#else
                corr_r[y][x] =  __builtin_amdgcn_sdot8(xr[x],yr[y], corr_r[y][x], false);
                corr_r[y][x] =  __builtin_amdgcn_sdot8(xi[x],yi[y], corr_r[y][x], false);
                corr_i[y][x] =  __builtin_amdgcn_sdot8(xr[x],yi[y], corr_i[y][x], false);
                corr_i[y][x] = -__builtin_amdgcn_sdot8(xi[x],yr[y],-corr_i[y][x], false);
#endif
            }
            //rotate data to the neighbour work items
            #pragma unroll
            for (int k=0; k<WI_SIZE; k++){
                xr[k] = __builtin_amdgcn_ds_bpermute(dest_x*4,xr[k]);
                xi[k] = __builtin_amdgcn_ds_bpermute(dest_x*4,xi[k]);
                yr[k] = __builtin_amdgcn_ds_bpermute(dest_y*4,yr[k]);
                yi[k] = __builtin_amdgcn_ds_bpermute(dest_y*4,yi[k]);
            }
        }
    }
    __global int *out=(corr_buf + 
                     (((FREQ_ID*NUM_BLOCKS + BLOCK_ID)*BLOCK_SIZE
                                           + yl*WI_SIZE)*BLOCK_SIZE
                                           + xl*WI_SIZE)*2);
    #pragma unroll
    for (int y=0; y<WI_SIZE; y++){
        #pragma unroll
        for (int x=0; x<WI_SIZE; x++) {
            atomic_add(out++,corr_i[y][x]); //ri-ir
            atomic_add(out++,corr_r[y][x]); //rr+ii
        }
        out+=(BLOCK_SIZE-WI_SIZE)*2;
    }
}
