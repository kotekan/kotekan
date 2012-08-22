#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <sys/time.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>

#include "kernel.h"

#define HI_NIBBLE(b) (((b) >> 4) & 0x0F)
#define LO_NIBBLE(b) ((b) & 0x0F)

#define N_ANT 256
#define N_ITER 128*1024


double e_time(void)
{
  static struct timeval now;
  gettimeofday(&now, NULL);
  return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

int main(int argc, char ** argv) {
  double cputime=0;

  unsigned char *data_block;
  data_block=malloc(N_ITER*N_ANT);
  //-------------------------------------------------------------- 
  //-------------------------------------------------------------- 
  // Read in test data
  FILE *fp;
  fp=fopen("block2.dat", "r");
  int ct=fread(data_block, 1, N_ITER*N_ANT, fp);
  if (ct) ct=0;
  fclose(fp);
  //-------------------------------------------------------------- 
  //-------------------------------------------------------------- 


  cl_int err;

// 1. Get a platform.
  cl_platform_id platform;
  clGetPlatformIDs( 1, &platform, NULL );


// 2. Find a gpu device.
  cl_device_id device;
  clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  cl_ulong lm;
  clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &lm, NULL);
  //printf("Local Mem: %i\n",lm);

  cl_uint mcl,mcm;
  clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &mcl, NULL);
  clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &mcm, NULL);
  float card_tflops = mcl*1e6 * mcm*16*4*2 / 1e12;

// 3. Create a context and command queue on that device.
  cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue( context, device, 0, NULL );


// 4. Perform runtime source compilation, and obtain kernel entry point.
  cl_program program = clCreateProgramWithSource( context, 1, &kern, NULL, &err );
  if (err) printf("Error in clCreateProgramWithSource: %i\n",err);

  err = clBuildProgram( program, 1, &device, NULL, NULL, NULL );
  if (err) printf("Error in clBuildProgram: %i\n",err);

  cl_kernel corr_kernel = clCreateKernel( program, "corr", &err );
  if (err) printf("Error in clCreateKernel: %i\n",err);

// 5. Create a data buffer.
  cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				       N_ITER * N_ANT, data_block, NULL );
  int s1_blk = 32;
  int n_blk = (N_ANT / s1_blk) * (N_ANT / s1_blk + 1) / 2.;
  int len=n_blk*(s1_blk*s1_blk)*2.;
  int *zeros=calloc(len,sizeof(cl_int));
  cl_mem corr_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
				      len * sizeof(cl_int), zeros, NULL );
  free(zeros);

// 6. Launch the kernel.
  unsigned int global_id_x_map[n_blk];
  unsigned int global_id_y_map[n_blk];
  for (int i=0; i<n_blk; i++)
  {
    int t = (sqrt(1 + 8*(n_blk-i-1))-1)/2;
    int y = N_ANT/s1_blk-t-1;
    int x = (t+1)*(t+2)/2 + (i - n_blk)+y;
    global_id_x_map[i] = x;
    global_id_y_map[i] = y;
  }
  cl_mem id_x_map = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				   n_blk * sizeof(cl_uint), global_id_x_map, &err);
  if (err) printf("Error in clCreateBuffer %i\n", err);
  cl_mem id_y_map = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
				   n_blk * sizeof(cl_uint), global_id_y_map, &err);
  if (err) printf("Error in clCreateBuffer %i\n", err);

  clSetKernelArg(corr_kernel, 0, sizeof(input_buffer), (void*) &input_buffer);
  clSetKernelArg(corr_kernel, 1, sizeof(corr_buffer), (void*) &corr_buffer);
  clSetKernelArg(corr_kernel, 2, sizeof(id_x_map), (void*) &id_x_map);
  clSetKernelArg(corr_kernel, 3, sizeof(id_y_map), (void*) &id_y_map);
  clSetKernelArg(corr_kernel, 4, 4*16 *4 * sizeof(cl_uint), NULL);

  int nkern=1000;
  unsigned int n_caccum=N_ITER/256;
  size_t gws_corr[3]={8,8,n_blk*n_caccum};
  size_t lws_corr[3]={8,8,1};
  for (int i=0; i<nkern; i++)
    err=clEnqueueNDRangeKernel(queue, corr_kernel, 3, NULL, gws_corr, lws_corr, 0, NULL, NULL);
  if (err) printf("Error enqueueing! %i\n",err);


  printf("Running %i iterations of full corr (%iK time samples, %i antennas)\n",
	 nkern, N_ITER/1024, N_ANT);
  cputime = e_time();
  clFinish(queue);
  cputime = e_time()-cputime;
  printf("Unpacking rate: %6.4fs on GPU (%.1f kHz)\n",cputime,N_ITER/cputime/1000*nkern);
  printf("    [Theoretical max: @%.1f TFLOPS, %.1f kHz; %2.0f%% efficiency]\n",
	 card_tflops, card_tflops*1e12 / (N_ANT/2.*(N_ANT+1.) * 2. * 2.) / 1e3,
	 100.*nkern*N_ITER/cputime / (card_tflops*1e12) * N_ANT/2.*(N_ANT+1.) * 2. * 2.);
  printf("    [Algorithm max:   @%.1f TFLOPS, %.1f kHz; %2.0f%% efficiency]\n",
	 card_tflops, card_tflops*1e12 / (n_blk * s1_blk * s1_blk * 2. * 2.) / 1e3,
	 100.*nkern*N_ITER/cputime / (card_tflops*1e12) * n_blk * s1_blk * s1_blk * 2. * 2.);

// 7. Look at the results via synchronous buffer map.
  int *corr_ptr;
  corr_ptr = clEnqueueMapBuffer(queue, corr_buffer, CL_TRUE, CL_MAP_READ, 0,
			   len*sizeof(cl_int), 0, NULL, NULL, &err);
  if (err) printf("Error in clEnqueueMapBuffer: %i\n",err);
  clFinish( queue );

  //-------------------------------------------------------------- 
  //-------------------------------------------------------------- 
  //Calculate correct answers
  int *corr_re, *corr_im;
  unsigned int *accum_re, *accum_im;
  int dat_x_re,dat_x_im,dat_y_re,dat_y_im;
  corr_re=malloc((N_ANT+1)*N_ANT/2*sizeof(int));
  corr_im=malloc((N_ANT+1)*N_ANT/2*sizeof(int));
  accum_re=malloc(N_ANT*sizeof(int));
  accum_im=malloc(N_ANT*sizeof(int));
  memset(corr_re,0,(N_ANT+1)*N_ANT/2*sizeof(int));
  memset(corr_im,0,(N_ANT+1)*N_ANT/2*sizeof(int));
  memset(accum_re,0,N_ANT*sizeof(int));
  memset(accum_im,0,N_ANT*sizeof(int));
  int idx;
  cputime = e_time();
  for (int i=0; i<N_ITER; i++)
  {
    int addr=i*N_ANT;
    idx=0;
    for (int ant_y=0; ant_y<N_ANT; ant_y++)
    { 
      //lookup and parse real and imag of A
      dat_y_re = LO_NIBBLE(data_block[addr+ant_y]);
      dat_y_im = HI_NIBBLE(data_block[addr+ant_y]);
      //accumulate antenna values
      accum_re[ant_y]+=dat_y_re;
      accum_im[ant_y]+=dat_y_im;
      for (int ant_x=ant_y; ant_x<N_ANT; ant_x++)
      {
	//lookup and parse real and imag of B
	dat_x_re = LO_NIBBLE(data_block[addr+ant_x]);
	dat_x_im = HI_NIBBLE(data_block[addr+ant_x]);
	//calculate B * A*
	corr_re[idx] += dat_y_re * dat_x_re + dat_y_im * dat_x_im;
	corr_im[idx] += dat_y_re * dat_x_im - dat_y_im * dat_x_re;
	idx++;
      }
    }
  }

  err=0;
  idx=0;
  for (int ant_y=0; ant_y < N_ANT; ant_y++)
  {
    for (int ant_x=ant_y; ant_x < N_ANT; ant_x++)
    {
      int nx=ant_x / s1_blk;
      int ny=ant_y / s1_blk;
      int blkid=0;
      for (int y=0; y<ny; y++) blkid+=(N_ANT/32-y);
      blkid+=nx-ny;
      int gpu_addr=blkid*s1_blk*s1_blk + (ant_y%32)*32 + (ant_x%32);
      int gpu_corr_re=corr_ptr[gpu_addr*2]/nkern;
      int gpu_corr_im=corr_ptr[gpu_addr*2+1]/nkern;
     // printf("%d %d\n",gpu_corr_re, gpu_corr_im);
      if (corr_re[idx] != gpu_corr_re || corr_im[idx] != gpu_corr_im) err++;
      dat_x_re = LO_NIBBLE(data_block[ant_y]);
      dat_x_im = HI_NIBBLE(data_block[ant_y]);
      idx++;
    }
  }
  if (err >= 1) printf("Error with correlation/accumulation!\n");
  else printf("Correlation/accumulation successful! CPU matches GPU.\n");

  cputime=e_time()-cputime;
  printf("Full Corr: %4.2fs on CPU (%.2f kHz)\n",cputime,N_ITER/cputime/1e3);
  free(corr_re);
  free(corr_im);
  free(accum_re);
  free(accum_im);
  //-------------------------------------------------------------- 
  //-------------------------------------------------------------- 

  free(data_block);

  err = clEnqueueUnmapMemObject(queue,corr_buffer,corr_ptr,0,NULL,NULL);
  if (err) printf("Error in clEnqueueUnmapMemObject!\n");
  clFinish(queue);

  clReleaseKernel(corr_kernel);
  clReleaseProgram(program);
  clReleaseMemObject(input_buffer);
  clReleaseMemObject(corr_buffer);
  clReleaseMemObject(id_x_map);
  clReleaseMemObject(id_y_map);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  return 0;
}
