#define TILE_DIM 32
#define BLOCK_ROWS 8


__kernel void transpose(__global float *input, __global float *output) {
  
  uint width1 = get_global_size(0);
  uint width2 = get_global_size(1)*4;

  __local float tile[TILE_DIM][TILE_DIM];  //32x32
  uint x  = get_group_id(0) * TILE_DIM + get_local_id(0);
  uint y =  get_group_id(1) * TILE_DIM + get_local_id(1);

#pragma unroll
  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS){
    tile[get_local_id(1)+j][get_local_id(0)] = input[(y+j)*width1 + x];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  x = get_group_id(1) * TILE_DIM + get_local_id(0);
  y = get_group_id(0) * TILE_DIM + get_local_id(1);

#pragma unroll
  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS){
    output[(y+j)*(width2+32) + x] = tile[get_local_id(0)][get_local_id(1)+j];
  }


}
