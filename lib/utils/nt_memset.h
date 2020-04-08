#ifndef NT_MEMSET
#define NT_MEMSET
#include <stdint.h> // for uint8_t
#include <stdlib.h> // for size_t

#ifdef __cplusplus
extern "C" {
#endif

void nt_memset(void* dest, uint8_t val, size_t len);

#ifdef __cplusplus
}
#endif

#endif
