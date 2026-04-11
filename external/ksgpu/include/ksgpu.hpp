#ifndef _KSGPU_HPP
#define _KSGPU_HPP

// By default, pybind11 files are not included!
//
// #include "ksgpu/dlpack.h"                // from https://github.com/dmlc/dlpack
// #include "ksgpu/pybind11.hpp"            // externally visible Array<> converters
// #include "ksgpu/pybind11_utils.hpp"      // utility functions, unlikely to be useful from outside ksgpu

// Array class
#include "ksgpu/Array.hpp"

// CpuThreadPool: run multiple CPU threads with dynamic load-balancing, intended for timing.
#include "ksgpu/CpuThreadPool.hpp"

// Dtype class (used in Array)
#include "ksgpu/Dtype.hpp"

// KernelTimer: simple header-only class for timing cuda kernels.
#include "ksgpu/KernelTimer.hpp"

// constexpr_is_pow2(), constexpr_ilog2(), etc.
#include "ksgpu/constexpr_functions.hpp"

// CUDA_CALL(), CUDA_PEEK(), CudaStreamWrapper
#include "ksgpu/cuda_utils.hpp"

// __device__ inline functions.
#include "ksgpu/device_fp16.hpp"
#include "ksgpu/device_mma.hpp"          // C++ wrappers for mma.* PTX instructions
#include "ksgpu/device_transposes.hpp"   // warp_transpose(), etc.

// af_alloc(), af_copy(), af_clone()
#include "ksgpu/mem_utils.hpp"

// Drop-in-replacement for 1d and 2d GPU-to-GPU cudaMemcpyAsync(), but uses SMs instead of a copy engine.
#include "ksgpu/memcpy_kernels.hpp"

// rand_int(), rand_uniform(), randomize()
#include "ksgpu/rand_utils.hpp"

// to_str(), from_str(), tuple_str(), type_name()
#include "ksgpu/string_utils.hpp"

// assert_arrays_equal()
#include "ksgpu/test_utils.hpp"

// get_time(), time_diff(), time_since()
#include "ksgpu/time_utils.hpp"

// xassert(), xassert_msg(), xassert_eq(), etc.
#include "ksgpu/xassert.hpp"

#endif // _KSGPU_HPP
