#define TILE_DIM 32
#define BLOCK_ROWS 8

__kernel void transpose(__global float *input, __global float *output) {
    uint nelems = get_global_size(0);
    uint ntimes = get_global_size(1) * TILE_DIM/BLOCK_ROWS;


    __local float tile[TILE_DIM][TILE_DIM+1];  //avoid LDS bank conflicts
    uint x =  get_group_id(0) * TILE_DIM + get_local_id(0);
    uint y =  get_group_id(1) * TILE_DIM + get_local_id(1);

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
        tile[get_local_id(1)+j][get_local_id(0)] = input[(y+j)*nelems + x];

    barrier(CLK_LOCAL_MEM_FENCE);
    x = get_group_id(1) * TILE_DIM + get_local_id(0);
    y = get_group_id(0) * TILE_DIM + get_local_id(1);

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
        output[(y+j)*(ntimes+64) + x] = tile[get_local_id(0)][get_local_id(1)+j];
}
