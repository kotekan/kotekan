ARCH =
ARCH += -gencode arch=compute_80,code=sm_80
ARCH += -gencode arch=compute_86,code=sm_86
ARCH += -gencode arch=compute_89,code=sm_89
# ARCH += -gencode arch=compute_90,code=sm_90

NVCC = nvcc -std=c++17 $(ARCH) -m64 -O3 --compiler-options -Wall,-fPIC
SHELL := /bin/bash

.DEFAULT_GOAL: all
.PHONY: all clean .FORCE

HFILES = \
  include/gputils/Array.hpp \
  include/gputils/Barrier.hpp \
  include/gputils/CpuThreadPool.hpp \
  include/gputils/CudaStreamPool.hpp \
  include/gputils/ThreadSafeRingBuffer.hpp \
  include/gputils/complex_type_traits.hpp \
  include/gputils/constexpr_functions.hpp \
  include/gputils/cuda_utils.hpp \
  include/gputils/device_mma.hpp \
  include/gputils/mem_utils.hpp \
  include/gputils/memcpy_kernels.hpp \
  include/gputils/rand_utils.hpp \
  include/gputils/string_utils.hpp \
  include/gputils/system_utils.hpp \
  include/gputils/test_utils.hpp \
  include/gputils/time_utils.hpp

OFILES = \
  src_lib/Array.o \
  src_lib/Barrier.o \
  src_lib/CpuThreadPool.o \
  src_lib/CudaStreamPool.o \
  src_lib/cuda_utils.o \
  src_lib/mem_utils.o \
  src_lib/memcpy_kernels.o \
  src_lib/rand_utils.o \
  src_lib/string_utils.o \
  src_lib/system_utils.o \
  src_lib/test_utils.o

LIBFILES = \
  lib/libgputils.a \
  lib/libgputils.so

XFILES = \
  bin/time-atomic-add \
  bin/time-fma \
  bin/time-l2-cache \
  bin/time-local-transpose \
  bin/time-memcpy-kernels \
  bin/time-shared-memory \
  bin/time-tensor-cores \
  bin/time-warp-shuffle \
  bin/scratch \
  bin/reverse-engineer-mma \
  bin/test-array \
  bin/test-memcpy-kernels \
  bin/test-sparse-mma \
  bin/show-devices

SRCDIRS = \
  include \
  include/gputils \
  src_bin \
  src_lib

all: $(LIBFILES) $(XFILES)

# Not part of 'make all', needs explicit 'make source_files.txt'
source_files.txt: .FORCE
	rm -f source_files.txt
	shopt -s nullglob && for d in $(SRCDIRS); do for f in $$d/*.cu $$d/*.hpp $$d/*.cuh; do echo $$f; done; done >$@

clean:
	rm -f $(XFILES) $(LIBFILES) source_files.txt *~
	shopt -s nullglob && for d in $(SRCDIRS); do rm -f $$d/*~ $$d/*.o; done

%.o: %.cu $(HFILES)
	$(NVCC) -c -o $@ $<

bin/%: src_bin/%.o lib/libgputils.a
	mkdir -p bin && $(NVCC) -o $@ $^

lib/libgputils.so: $(OFILES)
	@mkdir -p lib
	rm -f $@
	$(NVCC) -shared -o $@ $^

lib/libgputils.a: $(OFILES)
	@mkdir -p lib
	rm -f $@
	ar rcs $@ $^

INSTALL_DIR ?= /usr/local

install: $(LIBFILES)
	mkdir -p $(INSTALL_DIR)/include
	mkdir -p $(INSTALL_DIR)/lib
	cp -rv lib $(INSTALL_DIR)/
	cp -rv include $(INSTALL_DIR)/
