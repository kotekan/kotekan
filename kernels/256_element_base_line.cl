#pragma OPENCL EXTENSION cl_amd_printf : enable

//HARDCODE ALL OF THESE!
#define N_ANT  256u  //2560u
#define N_BLK   36u  //3240u // N(N+1)/2 where N=(N_ANT/32)
#define N_ANT_4 64u  //640u  // N/4



__kernel void corr(	__global uint *packed,
				__global  int *corr_buf,
				__global uint *id_x_map,
				__global uint *id_y_map,
				__local  uint *unpacked)
{
	//uint lx = get_local_id(0);
	//uint ly = get_local_id(1);

	//uint z = get_global_id(2) / N_BLK;
	//uint blkid = get_global_id(2) - z*N_BLK;
	uint x = id_x_map[get_global_id(2)%N_BLK];
	uint y = id_y_map[get_global_id(2)%N_BLK];

	//uint addr_x = ((get_local_size(0)*id_x_map[get_global_id(2)%N_BLK]+get_local_id(0))  + get_global_id(2) / N_BLK*N_ANT*64); // / 4;
	//uint addr_y = ((id_y_map[get_global_id(2)%N_BLK]*8 + get_global_id(1)) + get_global_id(2) / N_BLK*N_ANT*64 + get_local_id(0)*N_ANT_4); ///4;
	uint addr_x = ((get_local_size(0)*x+get_local_id(0))  + get_global_id(2) / N_BLK*N_ANT*64); // / 4;
	uint addr_y = ((y*8 + get_global_id(1)) + get_global_id(2) / N_BLK*N_ANT*64 + get_local_id(0)*N_ANT_4); ///4;


	uint4 corr_a=(0,0,0,0);
	//uint	 corr_a_s0 = 0;
	//uint  corr_a_s1 = 0;
	//uint  corr_a_s2 = 0;
	//uint  corr_a_s3 = 0;
	uint4 corr_b=(0,0,0,0);
	uint4 corr_c=(0,0,0,0);
	uint4 corr_d=(0,0,0,0);
	uint4 corr_e=(0,0,0,0);
	uint4 corr_f=(0,0,0,0);
	uint4 corr_g=(0,0,0,0);
	uint4 corr_h=(0,0,0,0);

	uint4 temp_u=(0,0,0,0);

	for (uint i=0; i<256; i+=get_local_size(0)){
		uint la=(get_local_id(0)*32 + get_local_id(1)*4);
		uint pa=packed[mad24(i,N_ANT_4,addr_y)];

		barrier(CLK_LOCAL_MEM_FENCE);
		unpacked[la]   = (((pa >>  0) & 0xf) << 16) + ((pa >>  4) & 0xf);
		unpacked[la+1] = (((pa >>  8) & 0xf) << 16) + ((pa >> 12) & 0xf);
		unpacked[la+2] = (((pa >> 16) & 0xf) << 16) + ((pa >> 20) & 0xf);
		unpacked[la+3] = (((pa >> 24) & 0xf) << 16) + ((pa >> 28) & 0xf);
		barrier(CLK_LOCAL_MEM_FENCE);

		for (uint j=0; j<get_local_size(0); j++){
			temp_u = vload4(j*8+get_local_id(1), unpacked);
			//uint temp_ua = unpacked[j*32+get_local_id(1)*4];
			//uint temp_ub = unpacked[j*32+get_local_id(1)*4+1];
			//uint temp_uc = unpacked[j*32+get_local_id(1)*4+2];
			//uint temp_ud = unpacked[j*32+get_local_id(1)*4+3];
			pa = packed[addr_x+(i+j)*N_ANT_4];

			//uint temp_par = (pa >>  0) & 0xf;
			//uint temp_pai = (pa >>  4) & 0xf;
			//uint temp_pbr = (pa >>  8) & 0xf;
			//uint temp_pbi = (pa >> 12) & 0xf;

			//uint temp_pcr = (pa >> 16) & 0xf;
			//uint temp_pci = (pa >> 20) & 0xf;
			//uint temp_pdr = (pa >> 24) & 0xf;
			//uint temp_pdi = (pa >> 28) & 0xf;

			corr_a.s0=mad24((pa >>  0) & 0xf, temp_u.s0, corr_a.s0);
			corr_a.s1=mad24((pa >>  4) & 0xf, temp_u.s0, corr_a.s1);
			corr_a.s2=mad24((pa >>  8) & 0xf, temp_u.s0, corr_a.s2);
			corr_a.s3=mad24((pa >> 12) & 0xf, temp_u.s0, corr_a.s3);

			corr_b.s0=mad24((pa >>  0) & 0xf, temp_u.s1, corr_b.s0);
			corr_b.s1=mad24((pa >>  4) & 0xf, temp_u.s1, corr_b.s1);
			corr_b.s2=mad24((pa >>  8) & 0xf, temp_u.s1, corr_b.s2);
			corr_b.s3=mad24((pa >> 12) & 0xf, temp_u.s1, corr_b.s3);

			corr_c.s0=mad24((pa >>  0) & 0xf, temp_u.s2, corr_c.s0);
			corr_c.s1=mad24((pa >>  4) & 0xf, temp_u.s2, corr_c.s1);
			corr_c.s2=mad24((pa >>  8) & 0xf, temp_u.s2, corr_c.s2);
			corr_c.s3=mad24((pa >> 12) & 0xf, temp_u.s2, corr_c.s3);

			corr_d.s0=mad24((pa >>  0) & 0xf, temp_u.s3, corr_d.s0);
			corr_d.s1=mad24((pa >>  4) & 0xf, temp_u.s3, corr_d.s1);
			corr_d.s2=mad24((pa >>  8) & 0xf, temp_u.s3, corr_d.s2);
			corr_d.s3=mad24((pa >> 12) & 0xf, temp_u.s3, corr_d.s3);


			corr_e.s0=mad24((pa >> 16) & 0xf, temp_u.s0, corr_e.s0);
			corr_e.s1=mad24((pa >> 20) & 0xf, temp_u.s0, corr_e.s1);
			corr_e.s2=mad24((pa >> 24) & 0xf, temp_u.s0, corr_e.s2);
			corr_e.s3=mad24((pa >> 28) & 0xf, temp_u.s0, corr_e.s3);

			corr_f.s0=mad24((pa >> 16) & 0xf, temp_u.s1, corr_f.s0);
			corr_f.s1=mad24((pa >> 20) & 0xf, temp_u.s1, corr_f.s1);
			corr_f.s2=mad24((pa >> 24) & 0xf, temp_u.s1, corr_f.s2);
			corr_f.s3=mad24((pa >> 28) & 0xf, temp_u.s1, corr_f.s3);

			corr_g.s0=mad24((pa >> 16) & 0xf, temp_u.s2, corr_g.s0);
			corr_g.s1=mad24((pa >> 20) & 0xf, temp_u.s2, corr_g.s1);
			corr_g.s2=mad24((pa >> 24) & 0xf, temp_u.s2, corr_g.s2);
			corr_g.s3=mad24((pa >> 28) & 0xf, temp_u.s2, corr_g.s3);

			corr_h.s0=mad24((pa >> 16) & 0xf, temp_u.s3, corr_h.s0);
			corr_h.s1=mad24((pa >> 20) & 0xf, temp_u.s3, corr_h.s1);
			corr_h.s2=mad24((pa >> 24) & 0xf, temp_u.s3, corr_h.s2);
			corr_h.s3=mad24((pa >> 28) & 0xf, temp_u.s3, corr_h.s3);
		}
	}

	uint addr_o = ((get_global_id(2)%N_BLK * 1024) + (get_global_id(1) * 4 * 32) + (get_local_id(0) * 4)) *2;

	atomic_add(&corr_buf[addr_o],   (corr_a.s0 >> 16) + (corr_a.s1 & 0xffff) );
	atomic_add(&corr_buf[addr_o+4], (corr_e.s0 >> 16) + (corr_e.s1 & 0xffff) );
	atomic_add(&corr_buf[addr_o+64], (corr_b.s0 >> 16) + (corr_b.s1 & 0xffff) );
	atomic_add(&corr_buf[addr_o+68], (corr_f.s0 >> 16) + (corr_f.s1 & 0xffff) );
	atomic_add(&corr_buf[addr_o+128], (corr_c.s0 >> 16) + (corr_c.s1 & 0xffff) );
	atomic_add(&corr_buf[addr_o+132], (corr_g.s0 >> 16) + (corr_g.s1 & 0xffff) );
	atomic_add(&corr_buf[addr_o+192], (corr_d.s0 >> 16) + (corr_d.s1 & 0xffff) );
	atomic_add(&corr_buf[addr_o+196], (corr_h.s0 >> 16) + (corr_h.s1 & 0xffff) );


	atomic_add(&corr_buf[addr_o+1], (corr_a.s1 >> 16) - (corr_a.s0 & 0xffff) );
	atomic_add(&corr_buf[addr_o+5], (corr_e.s1 >> 16) - (corr_e.s0 & 0xffff) );
	atomic_add(&corr_buf[addr_o+65], (corr_b.s1 >> 16) - (corr_b.s0 & 0xffff) );
	atomic_add(&corr_buf[addr_o+69], (corr_f.s1 >> 16) - (corr_f.s0 & 0xffff) );
	atomic_add(&corr_buf[addr_o+129], (corr_c.s1 >> 16) - (corr_c.s0 & 0xffff) );
	atomic_add(&corr_buf[addr_o+133], (corr_g.s1 >> 16) - (corr_g.s0 & 0xffff) );
	atomic_add(&corr_buf[addr_o+193], (corr_d.s1 >> 16) - (corr_d.s0 & 0xffff) );
	atomic_add(&corr_buf[addr_o+197], (corr_h.s1 >> 16) - (corr_h.s0 & 0xffff) );

	atomic_add(&corr_buf[addr_o+2], (corr_a.s2 >> 16) + (corr_a.s3 & 0xffff) );
	atomic_add(&corr_buf[addr_o+6], (corr_e.s2 >> 16) + (corr_e.s3 & 0xffff) );
	atomic_add(&corr_buf[addr_o+66], (corr_b.s2 >> 16) + (corr_b.s3 & 0xffff) );
	atomic_add(&corr_buf[addr_o+70], (corr_f.s2 >> 16) + (corr_f.s3 & 0xffff) );
	atomic_add(&corr_buf[addr_o+130], (corr_c.s2 >> 16) + (corr_c.s3 & 0xffff) );
	atomic_add(&corr_buf[addr_o+134], (corr_g.s2 >> 16) + (corr_g.s3 & 0xffff) );
	atomic_add(&corr_buf[addr_o+194], (corr_d.s2 >> 16) + (corr_d.s3 & 0xffff) );
	atomic_add(&corr_buf[addr_o+198], (corr_h.s2 >> 16) + (corr_h.s3 & 0xffff) );

	atomic_add(&corr_buf[addr_o+3], (corr_a.s3 >> 16) - (corr_a.s2 & 0xffff) );
  	atomic_add(&corr_buf[addr_o+7], (corr_e.s3 >> 16) - (corr_e.s2 & 0xffff) );
	atomic_add(&corr_buf[addr_o+67], (corr_b.s3 >> 16) - (corr_b.s2 & 0xffff) );
	atomic_add(&corr_buf[addr_o+71], (corr_f.s3 >> 16) - (corr_f.s2 & 0xffff) );
	atomic_add(&corr_buf[addr_o+131], (corr_c.s3 >> 16) - (corr_c.s2 & 0xffff) );
	atomic_add(&corr_buf[addr_o+135], (corr_g.s3 >> 16) - (corr_g.s2 & 0xffff) );
	atomic_add(&corr_buf[addr_o+195], (corr_d.s3 >> 16) - (corr_d.s2 & 0xffff) );
	atomic_add(&corr_buf[addr_o+199], (corr_h.s3 >> 16) - (corr_h.s2 & 0xffff) );
}
