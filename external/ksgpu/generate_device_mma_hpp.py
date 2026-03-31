#!/usr/bin/env python3
#
# Reference:
#   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma
#
# Note: I'm no longer running this as part of the build process (in an automated way).
#
# Instead, I'm keeping the output file (include/ksgpu/device_mma.hpp) in git, and occasionally
# updating by hand:
#
#   ./generate_device_mma_hpp.py > include/ksgpu/device_mma.hpp     


class Argument:
    def __init__(self, name, cuda_type, num_registers, is_const=False, pack=True, ptx_type=None, is_scalar=False, is_immediate=False):
        self.name = name
        self.cuda_type = cuda_type
        self.num_registers = num_registers
        self.is_const = is_const
        self.pack = pack
        self.ptx_type = ptx_type
        self.is_scalar = is_scalar
        self.is_immediate = is_immediate

        if is_immediate:
            assert is_const and is_scalar and (ptx_type is None)

        if is_scalar:
            assert (not pack) and (num_registers == 1)
        

    def make_template_arglist(self):
        return [ f'{self.cuda_type} {self.name}' ] if self.is_immediate else [ ]

    
    def make_cuda_arglist(self):
        amp = '' if self.is_const else '&'
        cv = 'const ' if self.is_const else ''
        
        if self.is_immediate:
            return [ ]
        elif self.is_scalar:
            return [ f'{self.cuda_type} {amp}{self.name}' ]
        elif self.pack:
            return [ f'{cv}{self.cuda_type} {self.name}[{self.num_registers}]' ]
        else:
            return [ f'{self.cuda_type} {cv}{self.name}{i}' for i in range(self.num_registers) ]


    def make_ptx_argstr(self, base):
        t = [ f'%{i}' for i in range(base, base + self.num_registers) ]
        t = ', '.join(t)
        if not self.is_scalar:
            t =  '{' + t + '}'
        return t


    def make_constraint_str(self):
        cv = 'const ' if self.is_const else ''
        constraint = '"r"' if self.is_const else '"=r"'

        if self.is_immediate:
            return f'"n" ({self.name})'

        if self.is_scalar:
            return f'{constraint} ({self.name})'
            
        ret = [ ]
        for i in range(self.num_registers):
            v = f'{self.name}[{i}]' if self.pack else f'{self.name}{i}'
            if self.ptx_type is not None:
                v = f'*({cv}{self.ptx_type} *) &{v}'
            ret.append(f'{constraint} ({v})')

        return ', '.join(ret)


####################################################################################################


def emit_kernel(cuda_name, ptx_name, *args):
    # 'ordered_metadata' was introduced in nvcc 12.5 (I think), and we accommodate it with a hack (see below).
    # FIXME some day, when nvcc 12.4 is ancient history, this hack can be removed.
    omstr = '::ordered_metadata'
    omi = ptx_name.find(omstr)
    
    template_arglist = [ ]
    cuda_arglist = [ ]
    
    for arg in args:
        template_arglist += arg.make_template_arglist()
        cuda_arglist += arg.make_cuda_arglist()

    template_argstr = ', '.join(template_arglist)
    cuda_argstr = ', '.join(cuda_arglist)
    
    print(f'')
    print(f'// D = A*B + C')

    if len(template_arglist) > 0:
        print(f'template<{template_argstr}>')
    
    print(f'__device__ __forceinline__')
    print(f'void {cuda_name}({cuda_argstr})')
    print(f'{{')

    if omi < 0:
        print(f'    asm("{ptx_name} "')
    else:
        ptx_name2 = ptx_name[:omi] + ptx_name[omi+len(omstr):]
        print(f'    asm(')
        print(f'#if CUDART_VERSION >= 12050')
        print(f'        "{ptx_name} "')
        print(f'#else')
        print(f'        "{ptx_name2} "')
        print(f'#endif')

    base = 0
    for i,arg in enumerate(args):
        s = f'"{arg.make_ptx_argstr(base)}'
        t = ', "' if (i < len(args)-1) else ';" :'
        base += arg.num_registers
        print(f'        {s}{t}')

    for i,arg in enumerate(args):
        t = ''
        if i == 0:
            t = ' :'
        elif i < len(args)-1:
            t = ','

        s = arg.make_constraint_str()
        print(f'        {s}{t}')
    
    print(f'    );')
    print(f'}}')
    print(f'')


####################################################################################################

    
def emit_dense_mma(cuda_name, ptx_name, cuda_type, dbits, sbits, m, n, k, ptx_type=None, s=1):
    # Register counts
    na = (m*k*s*sbits) // 1024
    nb = (k*n*s*sbits) // 1024
    nc = (m*n*s*dbits) // 1024

    emit_kernel(
        cuda_name,
        ptx_name,
        Argument('d', cuda_type, nc, ptx_type=ptx_type),
        Argument('a', cuda_type, na, ptx_type=ptx_type, is_const=True),
        Argument('b', cuda_type, nb, ptx_type=ptx_type, is_const=True),
        Argument('c', cuda_type, nc, ptx_type=ptx_type, is_const=True)
    )


def emit_dense_f16_mma(m, n, k, s=1, layout=None):
    """The 'layout' parameter is only used for m8n8k4, and is a string pair such as ('row','col')."""
    
    cuda_name = f'mma_f16_m{m}_n{n}_k{k}'
    if layout is not None:
        cuda_name = f'{cuda_name}_{layout[0][0]}{layout[1][0]}'

    a = 'row' if (layout is None) else layout[0]
    b = 'col' if (layout is None) else layout[1]
    ptx_name = f'mma.sync.aligned.m{m}n{n}k{k}.{a}.{b}.f16.f16.f16.f16'
    
    emit_dense_mma(cuda_name, ptx_name, '__half2', 16, 16, m, n, k, ptx_type='uint', s=s)


def emit_dense_int_mma(sbits, m, n, k):
    typename = f's{sbits}' if (sbits > 1) else 'b1'
    satfinite = '.satfinite' if (sbits > 1) else ''
    suffix = '' if (sbits > 1) else '.and.popc'
    cuda_name = f'mma_{typename}_m{m}_n{n}_k{k}'
    ptx_name = f'mma.sync.aligned.m{m}n{n}k{k}.row.col{satfinite}.s32.{typename}.{typename}.s32{suffix}'
    emit_dense_mma(cuda_name, ptx_name, 'int', 32, sbits, m, n, k)


def emit_sparse_f16_mma(m, n, k):
    cuda_name = f'mma_sp_f16_m{m}_n{n}_k{k}'
    ptx_name = f'mma.sp::ordered_metadata.sync.aligned.m{m}n{n}k{k}.row.col.f16.f16.f16.f16'

    # Register counts
    na = (m*k) // 128
    nb = (k*n) // 64
    nc = (m*n) // 64

    emit_kernel(
        cuda_name,
        ptx_name,
        Argument('d', '__half2', nc, ptx_type='uint'),
        Argument('a', '__half2', na, ptx_type='uint', is_const=True),
        Argument('b', '__half2', nb, ptx_type='uint', is_const=True),
        Argument('c', '__half2', nc, ptx_type='uint', is_const=True),
        Argument('e', 'uint', 1, pack=False, is_scalar=True, is_const=True),
        Argument('F', 'uint', 1, pack=False, is_scalar=True, is_const=True, is_immediate=True)
    )

    

####################################################################################################

    
if __name__ == '__main__':
    print(f'#ifndef _KSGPU_DEVICE_MMA_HPP')
    print(f'#define _KSGPU_DEVICE_MMA_HPP')
    print(f'')
    print(f'// Autogenerated by generate_device_mma_hpp.py')
    print(f'//')
    print(f'// Reference for matrix shapes:')
    print(f'//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-shape')
    print(f'//')
    print(f'// Reference for PTX instruction syntax:')
    print(f'//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma')
    print()
    print('#include <cuda_fp16.h>')
    print(f'')
    print(f'namespace ksgpu {{')
    print(f'')

    # float16
    emit_dense_f16_mma(16, 8, 8)
    emit_dense_f16_mma(16, 8, 16)

    # int4
    emit_dense_int_mma(4, 8, 8, 32)
    emit_dense_int_mma(4, 16, 8, 32)
    emit_dense_int_mma(4, 16, 8, 64)

    # int8
    emit_dense_int_mma(8, 8, 8, 16)
    emit_dense_int_mma(8, 16, 8, 16)
    emit_dense_int_mma(8, 16, 8, 32)

    # int1
    emit_dense_int_mma(1, 8, 8, 128)

    # sparse float16
    emit_sparse_f16_mma(16, 8, 16)
    emit_sparse_f16_mma(16, 8, 32)

    # The PTX ISA includes f16 m8n8k4 MMAs.
    # I tried generating wrappers for these, but timing showed that they were extremely slow.
    # I assume these MMAs are legacy instructions which are emulated on Ampere.
    # I left commented-out code here, and in ../generate_device_mma_hpp.py in case I ever want to revisit this.

    # for a in ['row','col']:
    #     for b in ['row','col']:
    #         emit_dense_f16_mma(8, 8, 4, s=4, layout=(a,b))
    
    print(f'')
    print(f'}} // namespace ksgpu')
    print(f'')
    print(f'#endif // _KSGPU_DEVICE_MMA_HPP')
