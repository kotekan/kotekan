#!/usr/bin/env python3

# Cloned from n2k/template_instantiations/make-instantiation.py.
# Filename syntax: dual_kernel_{nstations_total}_{nfreq}.cu

import re
import os
import sys

def uassert(cond):
    if not cond:
        print('usage: make-instantation.py dirname/dual_kernel_{nstations}_{nfreq}.cu',
              file=sys.stderr)
        exit(1)

uassert(len(sys.argv) == 2)

output_filename = sys.argv[1]
m = re.match(r'dual_kernel_(\d+)_(\d+)\.cu', os.path.basename(output_filename))
uassert(m is not None)

NS = int(m.group(1))
NF = int(m.group(2))

with open(output_filename, 'w') as fout:
    print('#include "../include/n2k_dual/internals/DualCorrelatorKernel.hpp"', file=fout)
    print('namespace {', file=fout)
    print('    #pragma nv_diag_suppress 177  // spurious nvcc warning "k declared but never referenced"', file=fout)
    print(f'    n2k_dual::DualCorrelatorKernel<{NS},{NF}> k;', file=fout)
    print('}', file=fout)
