# { Set to constant; Store to memory; loop }
#     Inputs: []
#     Outputs: [E, E1, E2, Esum, Esum_zero]
#         E::Int32
#             Cplx(0) => SIMD(2)
#             Dish(0) => SIMD(4)
#             Dish(1) => Thread(0)
#             Dish(2) => Thread(1)
#             Dish(3) => Thread(2)
#             Dish(4) => Thread(3)
#             Dish(5) => Thread(4)
#             Dish(6) => Warp(0)
#             Dish(7) => Warp(1)
#             Dish(8) => Warp(2)
#             Freq(0) => Block(0)
#             Freq(1) => Block(1)
#             Freq(2) => Block(2)
#             Freq(3) => Block(3)
#             Freq(4) => Block(4)
#             Freq(5) => Block(5)
#             Freq(6) => Block(6)
#             Freq(7) => Block(7)
#             Polr(0) => SIMD(3)
#         E1::Int32
#             Cplx(0) => Register(0)
#             Dish(0) => SIMD(4)
#             Dish(1) => Thread(0)
#             Dish(2) => Thread(1)
#             Dish(3) => Thread(2)
#             Dish(4) => Thread(3)
#             Dish(5) => Thread(4)
#             Dish(6) => Warp(0)
#             Dish(7) => Warp(1)
#             Dish(8) => Warp(2)
#             Freq(0) => Block(0)
#             Freq(1) => Block(1)
#             Freq(2) => Block(2)
#             Freq(3) => Block(3)
#             Freq(4) => Block(4)
#             Freq(5) => Block(5)
#             Freq(6) => Block(6)
#             Freq(7) => Block(7)
#             Polr(0) => SIMD(3)
#         E2::Int32
#             Cplx(0) => Register(0)
#             Dish(0) => Register(2)
#             Dish(1) => Thread(0)
#             Dish(2) => Thread(1)
#             Dish(3) => Thread(2)
#             Dish(4) => Thread(3)
#             Dish(5) => Thread(4)
#             Dish(6) => Warp(0)
#             Dish(7) => Warp(1)
#             Dish(8) => Warp(2)
#             Freq(0) => Block(0)
#             Freq(1) => Block(1)
#             Freq(2) => Block(2)
#             Freq(3) => Block(3)
#             Freq(4) => Block(4)
#             Freq(5) => Block(5)
#             Freq(6) => Block(6)
#             Freq(7) => Block(7)
#             Polr(0) => Register(1)
#         Esum::Float32
#             Cplx(0) => Register(0)
#             Dish(0) => Register(2)
#             Dish(1) => Thread(0)
#             Dish(2) => Thread(1)
#             Dish(3) => Thread(2)
#             Dish(4) => Thread(3)
#             Dish(5) => Thread(4)
#             Dish(6) => Warp(0)
#             Dish(7) => Warp(1)
#             Dish(8) => Warp(2)
#             Freq(0) => Block(0)
#             Freq(1) => Block(1)
#             Freq(2) => Block(2)
#             Freq(3) => Block(3)
#             Freq(4) => Block(4)
#             Freq(5) => Block(5)
#             Freq(6) => Block(6)
#             Freq(7) => Block(7)
#             Polr(0) => Register(1)
#         Esum_zero::Float32
#             Cplx(0) => Thread(0)
#             Dish(0) => Thread(2)
#             Dish(1) => Thread(3)
#             Dish(2) => Thread(4)
#             Dish(3) => Register(0)
#             Dish(4) => Register(1)
#             Dish(5) => Register(2)
#             Dish(6) => Warp(0)
#             Dish(7) => Warp(1)
#             Dish(8) => Warp(2)
#             Freq(0) => Block(0)
#             Freq(1) => Block(1)
#             Freq(2) => Block(2)
#             Freq(3) => Block(3)
#             Freq(4) => Block(4)
#             Freq(5) => Block(5)
#             Freq(6) => Block(6)
#             Freq(7) => Block(7)
#             Polr(0) => Thread(1)
#     Unused: []
begin
    begin
        Esum_zero_0 = 0.0f0::Float32
        Esum_zero_1 = 0.0f0::Float32
        Esum_zero_2 = 0.0f0::Float32
        Esum_zero_3 = 0.0f0::Float32
        Esum_zero_4 = 0.0f0::Float32
        Esum_zero_5 = 0.0f0::Float32
        Esum_zero_6 = 0.0f0::Float32
        Esum_zero_7 = 0.0f0::Float32
    end
    begin
        #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1369 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + (((threadIdx()).x - 1) % Int32) & 0x1f) + 0] = Esum_zero_0
        #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1369 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + (((threadIdx()).x - 1) % Int32) & 0x1f + 32) + 0] = Esum_zero_1
        #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1369 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + (((threadIdx()).x - 1) % Int32) & 0x1f + 64) + 0] = Esum_zero_2
        #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1369 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + (((threadIdx()).x - 1) % Int32) & 0x1f + 96) + 0] = Esum_zero_3
        #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1369 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + (((threadIdx()).x - 1) % Int32) & 0x1f + 128) + 0] = Esum_zero_4
        #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1369 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + (((threadIdx()).x - 1) % Int32) & 0x1f + 160) + 0] = Esum_zero_5
        #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1369 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + (((threadIdx()).x - 1) % Int32) & 0x1f + 192) + 0] = Esum_zero_6
        #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1369 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + (((threadIdx()).x - 1) % Int32) & 0x1f + 224) + 0] = Esum_zero_7
    end
    for loopIdx1 = Int32(0):32767
        E = #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1313 =# @inbounds(E_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x08 + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x05 + (((threadIdx()).x - 1) % Int32) & 0x1f)]::Int4x8)
        (E1_0, E1_1) = convert(NTuple{2, Int8x4}, E)
        begin
            (E2_0, E2_2, E2_4, E2_6) = convert(NTuple{4, Int32}, E1_0)
            (E2_1, E2_3, E2_5, E2_7) = convert(NTuple{4, Int32}, E1_1)
        end
        begin
            Esum_0 = Float32(E2_0)::Float32
            Esum_1 = Float32(E2_1)::Float32
            Esum_2 = Float32(E2_2)::Float32
            Esum_3 = Float32(E2_3)::Float32
            Esum_4 = Float32(E2_4)::Float32
            Esum_5 = Float32(E2_5)::Float32
            Esum_6 = Float32(E2_6)::Float32
            Esum_7 = Float32(E2_7)::Float32
        end
        begin
            #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1373 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + ((((threadIdx()).x - 1) % Int32) & 0x1f) << 0x03)] += Esum_0
            #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1373 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + ((((threadIdx()).x - 1) % Int32) & 0x1f) << 0x03 + 1)] += Esum_1
            #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1373 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + ((((threadIdx()).x - 1) % Int32) & 0x1f) << 0x03 + 2)] += Esum_2
            #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1373 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + ((((threadIdx()).x - 1) % Int32) & 0x1f) << 0x03 + 3)] += Esum_3
            #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1373 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + ((((threadIdx()).x - 1) % Int32) & 0x1f) << 0x03 + 4)] += Esum_4
            #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1373 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + ((((threadIdx()).x - 1) % Int32) & 0x1f) << 0x03 + 5)] += Esum_5
            #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1373 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + ((((threadIdx()).x - 1) % Int32) & 0x1f) << 0x03 + 6)] += Esum_6
            #= /home/eschnett/src/jl/GPUIndexSpaces.jl/src/GPUIndexSpaces.jl:1373 =# @inbounds Esum_global[1 + (((((blockIdx()).x - 1) % Int32) & 0xff) << 0x0b + ((((threadIdx()).y - 1) % Int32) & 0x07) << 0x08 + ((((threadIdx()).x - 1) % Int32) & 0x1f) << 0x03 + 7)] += Esum_7
        end
    end
end
