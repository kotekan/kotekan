#ifndef NT_MEMCPY
#define NT_MEMCPY
#include <stdlib.h>

void nt_memcpy(void * dest, void * src, size_t len);

void nt_aligned_memcpy(void * dest, void * src, size_t len);

void nt_unaligned_memcpy(void * dest, void * src, size_t len);

#endif