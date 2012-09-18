#include <stdio.h>
#include <sys/time.h>
#include <memory.h>
#include <pthread.h>
#include <malloc.h>


#define NITER 100
#define SHUF 4

double e_time(void)
{
  static struct timeval now;
  gettimeofday(&now, NULL);
  return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

struct thread_data{
  char *src;
  char *dst;
  int len;
  int str;
};

void *copySub(void *threadarg)
{
  struct thread_data *data = (struct thread_data *) threadarg;
  char *src=data->src;
  char *dst=data->dst;
  int len=data->len;
  int str=data->str;

  int i,dx;

  for (i=0; i<NITER; i++)
  {
    for (dx=0; dx<(len>>str); dx++)
      {memcpy(dst+(dx<<str),src+(dx<<str)*SHUF,1<<str);}
  }
}

int main(int argc, char **argv)
{
  int x=128*1024;
  int y=256;
  int i,j,dx,dy;
  int db, err;

  pthread_t cp_th[SHUF];
  struct thread_data thread_data_array[SHUF];

//  char *src=malloc(x*y*SHUF);
  char *src=memalign(1024*1024,x*y*SHUF);
  srand(time(NULL));
  for (i=0; i<x*y*SHUF; i++) {src[i] = rand();}

  char *dst[SHUF];
  for (i=0; i<SHUF; i++)
  {
//    dst[i]=malloc(x*y);
    dst[i]=memalign(1024*1024,x*y);
    memset(dst[i],'d'+i,x*y);
  }

  sleep(5);

  //loop over interleaving -- number of contiguous samples = 2^db
  for (db=5; db<=25; db++)
  {
    int ilv=1<<db;
    double cputime=0;

    cputime = e_time();
    for (i=0; i<SHUF; i++)
    {
      thread_data_array[i].src=src+i*ilv;
      thread_data_array[i].str=db;
      thread_data_array[i].dst=dst[i];
      thread_data_array[i].len=x*y;
      pthread_create(&cp_th[i], NULL, copySub, 
		     (void *)&thread_data_array[i]);
    }
    for (i=0; i<SHUF; i++){pthread_join(cp_th[i], NULL);}
    cputime=e_time()-cputime;


    printf("128k strided, buflen=%5i%s: %5.2fs (%5.1f Gbps)\n",
	   db<10?1<<db:1<<(db-10),db<10?" ":"K",
	   cputime, x*y*(SHUF*8 / (cputime/NITER) / 1e9));


    err=0;
    for (i=0; i<x*y/ilv; i++) {
      for (j=0; j<ilv; j++) {
	for (dx=0; dx<SHUF; dx++) {
	  if (src[(i*SHUF+dx)*ilv+j] !=  dst[dx][i*ilv+j]) {err++;}
	}
      }
    }
    if (err) {printf("Error with comparison! %i failures.\n",err);}
  }
  pthread_exit(NULL);
}

//gcc -O3 -ffast-math -msse2 -mfpmath=sse -mtune=corei7 -m64 main.c -lpthread
