#ifndef HSA_KOTEKAN_BASE_H
#define HSA_KOTEKAN_BASE_H

#ifdef __cplusplus
extern "C" {
#endif


#include "errors.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_finalize.h"
#include "hsa/hsa_ext_amd.h"

#include <assert.h>

void kotekan_hsa_start();

void * hsa_host_malloc(size_t len);

void hsa_host_free(void *ptr);

void kotekan_hsa_stop();

#ifdef __cplusplus
}
#endif

#endif
