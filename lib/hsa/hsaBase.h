#ifndef HSA_KOTEKAN_BASE_H
#define HSA_KOTEKAN_BASE_H

// Check HSA status code
#define HSA_CHECK(hsa_status)                   \
{                                               \
    if(hsa_status != HSA_STATUS_SUCCESS) {      \
      const char *_status;                       \
      hsa_status_string(hsa_status, &_status);   \
      ERROR("HSA_STATUS: %s", _status);          \
      assert(HSA_STATUS_SUCCESS == hsa_status); \
    }                                           \
}

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
