#ifndef NT_MEMSET
#define NT_MEMSET
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void nt_memset(void * dest, uint8_t val, size_t len);

#ifdef __cplusplus
}
#endif

#endif