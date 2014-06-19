#ifndef NT_MEMCPY
#define NT_MEMCPY
#include <stdlib.h>

void nt_memcpy(unsigned char * dest, unsigned char * src, size_t len);

void nt_aligned_memcpy(unsigned char* dest, unsigned char* src, size_t len);

void nt_unaligned_memcpy(unsigned char* dest, unsigned char* src, size_t len);

#endif