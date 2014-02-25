#include <stdio.h>
#include <sys/time.h>
#include <memory.h>

#define NITER 100

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
  int i,j,dx,dy;
  int db, err;

  char *src=malloc(x*y*4);
  char *dst=malloc(x*y);
  memset(src,'c',x*y*4);
  memset(dst,'d',x*y);

  sleep(5);

  //loop over interleaving -- number of contiguous samples = 2^db
  for (db=3; db<=25; db++)
  {
    int ilv=1<<db;
    /*
   //set up the input buffer
   srand(time(NULL));
   for (i=0; i<x*y/ilv; i++) {
     for (j=0; j<ilv; j++) {
       src[i*4*ilv+j] =  0*rand();
     }
   }*/

   double cputime=0;
   cputime = e_time();
   for (i=0; i<NITER; i++)
   {
     for (dx=0; dx<(x*y>>db); dx++)
       {memmove(dst+(dx<<db),src+(dx<<db)*4,1<<db);}
   }
   cputime=e_time()-cputime;
   printf("128k strided, buflen=%5i%s: %5.2fs (%5.1f Gbps)\n",
	  db<10?1<<db:1<<(db-10),db<10?" ":"K",
	  cputime, x*y*8 / (cputime/NITER) / 1e9);

   err=0;
   for (i=0; i<x*y/ilv; i++) {
     for (j=0; j<ilv; j++) {
       if (src[i*4*ilv+j] !=  dst[i*ilv+j]) {err++;}
     }
   }
   if (err) {printf("Error with comparison! %i failures.\n",err);}
   }
}

