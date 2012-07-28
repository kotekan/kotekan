#pragma OPENCL EXTENSION cl_amd_printf : enable

#define N_ANT 256

// A simple unpack kernel

__kernel void unpack(__global uint *input,
		     __global uint *repacked,
		     __global uint *accum,
		     uint nloop)
{
 uint x = get_global_id(0);
 uint y = get_global_id(1)*nloop;
 uint up[8];
 uint accum_local[8];
 uint addr_x = x*8u;

 #pragma unwrap 8
 for (uint j=0; j < 8; j++) accum_local[j]=0;

 uint addr_i = mad24(y, 64u, x);
 uint addr_o = mul24(y, 512u);
 uint addr_r = mul24(y, 256u);

 for (uint i=0; i < nloop; i++)
 {
  uint addr_o = addr_i * 8u;
  uint addr_r = addr_i * 4u;

  #pragma unroll 8
  for (uint j=0; j<8; j++) up[j] = (input[addr_i] >> (j*4)) & 0xf;
  #pragma unroll 4
  for (uint j=0; j<4; j++) repacked[addr_r + j] = mad24(up[2*j],0x10000u,up[2*j+1]);
  #pragma unroll 8
  for (uint j=0; j<8; j++) accum_local[j] += up[j];

  addr_i+=64;
 }
 #pragma unroll 8
 for (uint j=0; j<8; j++) atomic_add(&accum[addr_x + j], accum_local[j]);
}



__kernel //__attribute__((reqd_work_group_size(16,4,1)))
         void corr(__global uint *packed,
//	           __global uint *repacked,
	           __global int *corr_buf,
		   __const uint nblk,
		   __global uint *id_x_map,
		   __global uint *id_y_map,
		   __local uint *unpacked)
{
  uint lx = get_local_id(0);
  uint ly = get_local_id(1);


  uint z = get_global_id(2) / nblk;
  uint blkid = get_global_id(2) - z*nblk;
  uint x = id_x_map[get_global_id(2)-z*nblk];
  uint y = id_y_map[get_global_id(2)-z*nblk];

  uint q = (lx % 2) * 16;
  uint addr_x = ((get_local_size(0)*x+lx)*2  + z*N_ANT*256) / 4;
  uint addr_y = ((y*32 + get_global_id(1)*4) + z*N_ANT*256 + lx*N_ANT)/4;

  uint corr_a[4];
  uint corr_b[4];
  uint corr_c[4];
  uint corr_d[4];

  for (uint j=0; j< 4; j++) corr_a[j]=0;
  for (uint j=0; j< 4; j++) corr_b[j]=0;
  for (uint j=0; j< 4; j++) corr_c[j]=0;
  for (uint j=0; j< 4; j++) corr_d[j]=0;

  for (uint i=0; i<256; i+=get_local_size(0))
  {
    uint la=(lx*16 + ly*4);
    uint pa=packed[mad24(i,64u,addr_y)];

    unpacked[la]   = (((pa >>  0) & 0xf) << 16) + ((pa >>  4) & 0xf);
    unpacked[la+1] = (((pa >>  8) & 0xf) << 16) + ((pa >> 12) & 0xf);
    unpacked[la+2] = (((pa >> 16) & 0xf) << 16) + ((pa >> 20) & 0xf);
    unpacked[la+3] = (((pa >> 24) & 0xf) << 16) + ((pa >> 28) & 0xf);

    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint j=0; j<get_local_size(0); j++)
    {
      uint temp_ua = unpacked[j*16+ly*4];
      uint temp_ub = unpacked[j*16+ly*4+1];
      uint temp_uc = unpacked[j*16+ly*4+2];
      uint temp_ud = unpacked[j*16+ly*4+3];

      uint temp_par = (packed[addr_x+(i+j)*64] >> (q +  0)) & 0xf;
      uint temp_pai = (packed[addr_x+(i+j)*64] >> (q +  4)) & 0xf;
      uint temp_pbr = (packed[addr_x+(i+j)*64] >> (q +  8)) & 0xf;
      uint temp_pbi = (packed[addr_x+(i+j)*64] >> (q + 12)) & 0xf;

      corr_a[0]=mad24(temp_par, temp_ua, corr_a[0]);
      corr_a[1]=mad24(temp_pai, temp_ua, corr_a[1]);
      corr_a[2]=mad24(temp_pbr, temp_ua, corr_a[2]);
      corr_a[3]=mad24(temp_pbi, temp_ua, corr_a[3]);

      corr_b[0]=mad24(temp_par, temp_ub, corr_b[0]);
      corr_b[1]=mad24(temp_pai, temp_ub, corr_b[1]);
      corr_b[2]=mad24(temp_pbr, temp_ub, corr_b[2]);
      corr_b[3]=mad24(temp_pbi, temp_ub, corr_b[3]);

      corr_c[0]=mad24(temp_par, temp_uc, corr_c[0]);
      corr_c[1]=mad24(temp_pai, temp_uc, corr_c[1]);
      corr_c[2]=mad24(temp_pbr, temp_uc, corr_c[2]);
      corr_c[3]=mad24(temp_pbi, temp_uc, corr_c[3]);

      corr_d[0]=mad24(temp_par, temp_ud, corr_d[0]);
      corr_d[1]=mad24(temp_pai, temp_ud, corr_d[1]);
      corr_d[2]=mad24(temp_pbr, temp_ud, corr_d[2]);
      corr_d[3]=mad24(temp_pbi, temp_ud, corr_d[3]);
    }
  }

  uint addr_o = ((blkid * 1024) + (get_global_id(1) * 4 * 32) + (lx * 2)) * 2;

  atomic_add(&corr_buf[addr_o],   (corr_a[0] >> 16) + (corr_a[1] & 0xffff) );
  atomic_add(&corr_buf[addr_o+1], (corr_a[1] >> 16) - (corr_a[0] & 0xffff) );
  atomic_add(&corr_buf[addr_o+2], (corr_a[2] >> 16) + (corr_a[3] & 0xffff) );
  atomic_add(&corr_buf[addr_o+3], (corr_a[3] >> 16) - (corr_a[2] & 0xffff) );

  atomic_add(&corr_buf[addr_o+64], (corr_b[0] >> 16) + (corr_b[1] & 0xffff) );
  atomic_add(&corr_buf[addr_o+65], (corr_b[1] >> 16) - (corr_b[0] & 0xffff) );
  atomic_add(&corr_buf[addr_o+66], (corr_b[2] >> 16) + (corr_b[3] & 0xffff) );
  atomic_add(&corr_buf[addr_o+67], (corr_b[3] >> 16) - (corr_b[2] & 0xffff) );

  atomic_add(&corr_buf[addr_o+128], (corr_c[0] >> 16) + (corr_c[1] & 0xffff) );
  atomic_add(&corr_buf[addr_o+129], (corr_c[1] >> 16) - (corr_c[0] & 0xffff) );
  atomic_add(&corr_buf[addr_o+130], (corr_c[2] >> 16) + (corr_c[3] & 0xffff) );
  atomic_add(&corr_buf[addr_o+131], (corr_c[3] >> 16) - (corr_c[2] & 0xffff) );

  atomic_add(&corr_buf[addr_o+192], (corr_d[0] >> 16) + (corr_d[1] & 0xffff) );
  atomic_add(&corr_buf[addr_o+193], (corr_d[1] >> 16) - (corr_d[0] & 0xffff) );
  atomic_add(&corr_buf[addr_o+194], (corr_d[2] >> 16) + (corr_d[3] & 0xffff) );
  atomic_add(&corr_buf[addr_o+195], (corr_d[3] >> 16) - (corr_d[2] & 0xffff) );
}
