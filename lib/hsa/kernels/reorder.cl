//512 reordering index
//256 WI

__kernel void reorder(__global uint *data, __global uint *Re, __global uint *output){

  __local uint local_data[512];
  __local uint local_data2[512];
  uint local_address = get_local_id(0);

  local_data[local_address] = data[get_global_id(1)*512 + local_address];
  local_data[256 + local_address] = data[get_global_id(1)*512 +256+local_address];

  barrier(CLK_LOCAL_MEM_FENCE);

  //Reorder to natural order of time-pol-EW-NS
  local_data2[local_address] = local_data[ Re[local_address]];
  local_data2[256+local_address] = local_data[ Re[256+local_address]];

  barrier(CLK_LOCAL_MEM_FENCE);

  output[get_group_id(1)*512 + local_address] = local_data2[local_address];
  output[get_group_id(1)*512 +256+ local_address] = local_data2[256+local_address];

}