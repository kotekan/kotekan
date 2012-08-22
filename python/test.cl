#pragma OPENCL EXTENSION cl_amd_printf : enable

//HARDCODE ALL OF THESE!
#define N_ANT  256u  //2560u
#define N_BLK   36u  //3240u // N(N+1)/2 where N=(N_ANT/32)
#define N_ANT_4 64u  //640u  // N/4



__kernel void corr(__global uint *packed,
	           __global  int *corr_buf,
		   __global uint *id_x_map,
		   __global uint *id_y_map,
		   __local  uint *unpacked)
{
  uint lx = get_local_id(0);
  uint ly = get_local_id(1);

  uint z = get_global_id(2) / N_BLK;
  uint blkid = get_global_id(2) - z*N_BLK;
  uint x = id_x_map[get_global_id(2)-z*N_BLK];
  uint y = id_y_map[get_global_id(2)-z*N_BLK];

  uint addr_x = ((get_local_size(0)*x+lx)*4  + z*N_ANT*256) / 4;
  uint addr_y = ((y*32 + get_global_id(1)*4) + z*N_ANT*256 + lx*N_ANT)/4;

  //if ((lx==0) && (ly==0)) {printf("%i %i | %i %i\n",lx,ly,x,y);}

  uint corr_a[4]={0,0,0,0};
  uint corr_b[4]={0,0,0,0};
  uint corr_c[4]={0,0,0,0};
  uint corr_d[4]={0,0,0,0};
  uint corr_e[4]={0,0,0,0};
  uint corr_f[4]={0,0,0,0};
  uint corr_g[4]={0,0,0,0};
  uint corr_h[4]={0,0,0,0};

  for (uint i=0; i<256; i+=get_local_size(0))
  {
    uint la=(lx*32 + ly*4);
    uint pa=packed[mad24(i,N_ANT_4,addr_y)];

    barrier(CLK_LOCAL_MEM_FENCE);
    unpacked[la]   = (((pa >>  0) & 0xf) << 16) + ((pa >>  4) & 0xf);
    unpacked[la+1] = (((pa >>  8) & 0xf) << 16) + ((pa >> 12) & 0xf);
    unpacked[la+2] = (((pa >> 16) & 0xf) << 16) + ((pa >> 20) & 0xf);
    unpacked[la+3] = (((pa >> 24) & 0xf) << 16) + ((pa >> 28) & 0xf);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint j=0; j<get_local_size(0); j++)
    {
      uint temp_ua = unpacked[j*32+ly*4];
      uint temp_ub = unpacked[j*32+ly*4+1];
      uint temp_uc = unpacked[j*32+ly*4+2];
      uint temp_ud = unpacked[j*32+ly*4+3];

      uint temp_par = (packed[addr_x+(i+j)*N_ANT_4] >>  0) & 0xf;
      uint temp_pai = (packed[addr_x+(i+j)*N_ANT_4] >>  4) & 0xf;
      uint temp_pbr = (packed[addr_x+(i+j)*N_ANT_4] >>  8) & 0xf;
      uint temp_pbi = (packed[addr_x+(i+j)*N_ANT_4] >> 12) & 0xf;

      uint temp_pcr = (packed[addr_x+(i+j)*N_ANT_4] >> 16) & 0xf;
      uint temp_pci = (packed[addr_x+(i+j)*N_ANT_4] >> 20) & 0xf;
      uint temp_pdr = (packed[addr_x+(i+j)*N_ANT_4] >> 24) & 0xf;
      uint temp_pdi = (packed[addr_x+(i+j)*N_ANT_4] >> 28) & 0xf;

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


      corr_e[0]=mad24(temp_pcr, temp_ua, corr_e[0]);
      corr_e[1]=mad24(temp_pci, temp_ua, corr_e[1]);
      corr_e[2]=mad24(temp_pdr, temp_ua, corr_e[2]);
      corr_e[3]=mad24(temp_pdi, temp_ua, corr_e[3]);

      corr_f[0]=mad24(temp_pcr, temp_ub, corr_f[0]);
      corr_f[1]=mad24(temp_pci, temp_ub, corr_f[1]);
      corr_f[2]=mad24(temp_pdr, temp_ub, corr_f[2]);
      corr_f[3]=mad24(temp_pdi, temp_ub, corr_f[3]);

      corr_g[0]=mad24(temp_pcr, temp_uc, corr_g[0]);
      corr_g[1]=mad24(temp_pci, temp_uc, corr_g[1]);
      corr_g[2]=mad24(temp_pdr, temp_uc, corr_g[2]);
      corr_g[3]=mad24(temp_pdi, temp_uc, corr_g[3]);

      corr_h[0]=mad24(temp_pcr, temp_ud, corr_h[0]);
      corr_h[1]=mad24(temp_pci, temp_ud, corr_h[1]);
      corr_h[2]=mad24(temp_pdr, temp_ud, corr_h[2]);
      corr_h[3]=mad24(temp_pdi, temp_ud, corr_h[3]);
    }
  }

  uint addr_o = ((blkid * 1024) + (get_global_id(1) * 4 * 32) + (lx * 4)) * 2;

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


  atomic_add(&corr_buf[addr_o+4], (corr_e[0] >> 16) + (corr_e[1] & 0xffff) );
  atomic_add(&corr_buf[addr_o+5], (corr_e[1] >> 16) - (corr_e[0] & 0xffff) );
  atomic_add(&corr_buf[addr_o+6], (corr_e[2] >> 16) + (corr_e[3] & 0xffff) );
  atomic_add(&corr_buf[addr_o+7], (corr_e[3] >> 16) - (corr_e[2] & 0xffff) );

  atomic_add(&corr_buf[addr_o+68], (corr_f[0] >> 16) + (corr_f[1] & 0xffff) );
  atomic_add(&corr_buf[addr_o+69], (corr_f[1] >> 16) - (corr_f[0] & 0xffff) );
  atomic_add(&corr_buf[addr_o+70], (corr_f[2] >> 16) + (corr_f[3] & 0xffff) );
  atomic_add(&corr_buf[addr_o+71], (corr_f[3] >> 16) - (corr_f[2] & 0xffff) );

  atomic_add(&corr_buf[addr_o+132], (corr_g[0] >> 16) + (corr_g[1] & 0xffff) );
  atomic_add(&corr_buf[addr_o+133], (corr_g[1] >> 16) - (corr_g[0] & 0xffff) );
  atomic_add(&corr_buf[addr_o+134], (corr_g[2] >> 16) + (corr_g[3] & 0xffff) );
  atomic_add(&corr_buf[addr_o+135], (corr_g[3] >> 16) - (corr_g[2] & 0xffff) );

  atomic_add(&corr_buf[addr_o+196], (corr_h[0] >> 16) + (corr_h[1] & 0xffff) );
  atomic_add(&corr_buf[addr_o+197], (corr_h[1] >> 16) - (corr_h[0] & 0xffff) );
  atomic_add(&corr_buf[addr_o+198], (corr_h[2] >> 16) + (corr_h[3] & 0xffff) );
  atomic_add(&corr_buf[addr_o+199], (corr_h[3] >> 16) - (corr_h[2] & 0xffff) );
}
