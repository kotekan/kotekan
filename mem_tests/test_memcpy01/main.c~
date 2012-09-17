#include <stdio.h>
#include <sys/time.h>
#include <memory.h>

#include "ippi.h"
#include "ippm.h"
#include "mkl.h"


double e_time(void)
{
  static struct timeval now;
  gettimeofday(&now, NULL);
  return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}


int main(int argc, char **argv)
{
  int x=128*1024;
  int y=256;
  int i,dx,dy;
  int niter=1000;
  int db=25;

  /*
  Ipp32f *src = malloc(x*y);
  Ipp32f *dst = malloc(x*y);
  IppiSize srcRoi = { x,y };
  for (i=0; i<niter; i++)
    {ippiTranspose_8u_C1R ( src, x, dst, y, srcRoi );}
    {ippmTranspose_m_32f ( src, x, 4, x/4, y, dst, y, 4 );}
  */
  char *src=malloc(x*y*4);
  char *dst=malloc(x*y);

  double cputime=0;
  cputime = e_time();
  for (i=0; i<niter; i++)
  {
    for (dx=0; dx<(x*y>>db); dx++)
      {memcpy(dst+(dx<<db),src+(dx<<db)*4,2<<db);}
  }
    printf("%i\n",2<<2);
//  mkl_somatcopy('r', 'n', 1,1, 1.,src, 1, dst, 1);


  cputime=e_time()-cputime;
  printf("128k transpose: %4.2fs (%5.1f Gbps)\n", cputime, x*y*8 / (cputime/niter) / 1e9);
}

// gcc main.c -I/opt/intel/ipp/include -I/opt/intel/mkl/include -L/opt/intel/ipp/lib/intel64 -lippiu8 -lippm -L/opt/intel/lib/intel64 -lippcore -liomp5 -l

// gcc main.c -I/opt/intel/ipp/include -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -Wl,-R/usr/global/intel/mkl/10.2.6.038/lib/em64t -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_sequential -lm -lgomp
