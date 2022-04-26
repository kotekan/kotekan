# Creating the files in this directory

The current kernel is called `average`. This is just a placeholder, to
be replaced by the actual `frb` kernel.

- `average.jl`: Generated Julia code, output by the code generator
- `average.llvm`: LLVM code, output by the code generator
- `average.ptx`: PTX code, output by the code generator
- `average.sass`: SASS (Shader ASSembly code), output by the code generator
- `average.cubin`: Generated via
  `ptxas -arch=sm_86 -m64 average.ptx -o average.cubin`
- `average.fatbin`, `average.fatbin.c`: Generated via
  `fatbinary --create=average.fatbin -64 --cicc-cmdline='-ftz=1 -prec_div=1 -prec_sqrt=1 -fmad=1' --image3=kind=elf,sm=86,file=average.cubin --image3=kind=ptx,sm=86,file=average.ptx --embedded-fatbin=average.fatbin.c`
