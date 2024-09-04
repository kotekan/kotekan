# CHORD transpose kernel

using CUDA
using CUDASIMDTypes
using IndexSpaces
using Mustache

const Memory = IndexSpaces.Memory

const card = "A40"

if CUDA.functional()
    println("[Choosing CUDA device...]")
    CUDA.device!(0)
    println(name(device()))
    @assert name(device()) == "NVIDIA $card"
end

idiv(i::Integer, j::Integer) = (@assert iszero(i % j); i ÷ j)
ilog2(i::Integer) = (j = round(Int, log2(i)); @assert 2^j == i; j)

function shrink(value::Integer)
    typemin(Int32) <= value <= typemax(Int32) && return Int32(value)
    typemin(Int64) <= value <= typemax(Int64) && return Int64(value)
    return value
end
function shrinkmul(x::Integer, y::Symbol, ymax::Integer)
    @assert x >= 0 && ymax >= 0
    # We assume 0 <= y < ymax
    if x * (ymax - 1) <= typemax(Int32)
        # We can use 32-bit arithmetic        
        return :($(Int32(x)) * $y)
    elseif x * (ymax - 1) <= typemax(Int64)
        # We need to use 64-bit arithmetic        
        return :($(Int64(x)) * $y)
    else
        # Something is wrong
        @assert false
    end
end

################################################################################

@enum CHORDTag CplxTag DishTag PolrTag FreqTag TimeTag ThreadTag WarpTag BlockTag

const Cplx = Index{Physics,CplxTag} # complex numbers
const Dish = Index{Physics,DishTag} # dish
const Polr = Index{Physics,PolrTag} # polarization
const Freq = Index{Physics,FreqTag} # frequency
const Time = Index{Physics,TimeTag} # time

# Setup

@assert D * P == 2048
const T1 = 32

const W = 16                    # warps
const B = F                     # blocks

const Tloop = idiv(T, T1)

const num_simd_bits = 32
const num_threads = 32
const num_warps = W
const num_blocks = B
const num_blocks_per_sm = 1

const cacheline_length = 32     # counting in UInt32
const shmem_bytes = 65536

const kernel_setup = KernelSetup(num_threads, num_warps, num_blocks, num_blocks_per_sm, shmem_bytes)

################################################################################

function expand_indices(indices::AbstractVector{<:Index})
    new_indices = eltype(indices)[]
    for index in indices
        @assert ispow2(index.length)
        for bit in 0:(ilog2(index.length) - 1)
            push!(new_indices, typeof(index)(index.name, index.offset * 2^bit, 2))
        end
    end
    return new_indices
end

function match_indices(
    physics_indices::AbstractVector{<:Index{Tag1}},
    machine_indices::AbstractVector{<:Index{Tag2}},
    dict::Dict{Index{Tag1},Index{Tag2}}=Dict{Index{Tag1},Index{Tag2}}(),
) where {Tag1,Tag2}
    physics_indices = expand_indices(physics_indices)
    machine_indices = expand_indices(machine_indices)
    @assert length(physics_indices) <= length(machine_indices)
    dict = copy(dict)
    for (phys_ind, mach_ind) in zip(physics_indices, machine_indices)
        dict[phys_ind] = mach_ind
    end
    return Layout(dict)
end

################################################################################

# Here "dish" means "dish or polr". "time" means "time or freq"

# Phys     Glob    Reg       |   Reg       Shared   Phys
#                            |             
# dish0    mbyte0  simd0     |   simd0     sbyte0    time0
# dish1    mbyte1  simd1     |   simd1     sbyte1    time1
# dish2    mbyte2  reg0      |   thread2   bank0     time2
# dish3    mbyte3  reg1      |   thread3   bank1     time3
# dish4    mbyte4  thread0   |   thread4   bank2     time4
# dish5    mbyte5  thread1   |   thread0   bank3     dish0 
# dish6    mbyte6  thread2   |   thread1   bank4     dish1 
# dish7    line0   warp0     |   reg0      shared0   dish2 
# dish8    line1   warp1     |   reg1      shared1   dish3 
# dish9    line2   warp2     |   reg2      shared2   dish4 
# dish10   line3   warp3     |   reg3      shared3   dish5 
# time0    line4   reg2      |   reg4      shared4   dish6 
# time1    line5   reg3      |   warp0     shared5   dish7 
# time2    line6   reg4      |   warp1     shared6   dish8 
# time3    line7   thread3   |   warp2     shared7   dish9 
# time4    line8   thread4   |   warp3     shared8   dish10
# 
# Swaps necessary:
#     dish0/simd0   <-> time0/reg2
#     dish1/simd1   <-> time1/reg3
#     dish4/thread0 <-> dish0/reg2
#     dish5/thread1 <-> dish1/reg3
#     dish6/thread2 <-> dish1/reg4

################################################################################

const layout_E_memory = let
    physics_indices = Index{Physics}[
        IntValue(:intvalue, 1, 4), Cplx(:cplx, 1, C), Dish(:dish, 1, D), Polr(:polr, 1, P), Freq(:freq, 1, F), Time(:time, 1, T)
    ]
    machine_indices = Index{Machine}[SIMD(:simd, 1, 32), Memory(:memory, 1, 2^55)]
    match_indices(physics_indices, machine_indices)
end

const layout_E_registers = let
    physics_indices = Index{Physics}[
        IntValue(:intvalue, 1, 4), Cplx(:cplx, 1, C), Dish(:dish, 1, D), Polr(:polr, 1, P), Time(:time, 1, T), Freq(:freq, 1, F)
    ]
    machine_indices = Index{Machine}[
        SIMD(:simd, 1, 32),
        Register(:register, 1, 4),
        Thread(:thread, 1, 8),
        Warp(:warp, 1, 16),
        Register(:register, 4, 8),
        Thread(:thread, 8, 4),
        Loop(:time_loop, T1, Tloop),
        Block(:block, 1, B),
    ]
    match_indices(physics_indices, machine_indices)
end

const layout_E_shared = let
    physics_indices = Index{Physics}[
        IntValue(:intvalue, 1, 4),
        Cplx(:cplx, 1, C),
        Time(:time, 1, T1),
        Dish(:dish, 1, D),
        Polr(:polr, 1, P),
        Time(:time, T1, idiv(T, T1)),
        Freq(:freq, 1, F),
    ]
    machine_indices = Index{Machine}[
        SIMD(:simd, 1, 32), Shared(:shared, 1, 16384), Loop(:time_loop, T1, Tloop), Block(:block, 1, B)
    ]
    match_indices(physics_indices, machine_indices)
end

const layout_scatter_indices_memory = let
    physics_indices = Index{Physics}[IntValue(:intvalue, 1, 32), Dish(:dish, 1, D), Polr(:polr, 1, P)]
    machine_indices = Index{Machine}[SIMD(:simd, 1, 32), Memory(:memory, 1, 2^55)]
    match_indices(physics_indices, machine_indices)
end

const layout_scatter_indices_registers = let
    physics_indices = Index{Physics}[IntValue(:intvalue, 1, 32), Dish(:dish, 1, D), Polr(:polr, 1, P)]
    # This must match the E register mapping when writing to shared memory
    machine_indices = Index{Machine}[SIMD(:simd, 1, 32), Thread(:thread, 1, 4), Register(:register, 1, 32), Warp(:warp, 1, 16)]
    match_indices(physics_indices, machine_indices)
end

const layout_info_memory = let
    physics_indices = Index{Physics}[
        IntValue(:intvalue, 1, 32),
        Index{Physics,ThreadTag}(:thread, 1, num_threads),
        Index{Physics,WarpTag}(:warp, 1, num_warps),
        Index{Physics,BlockTag}(:block, 1, num_blocks),
    ]
    machine_indices = Index{Machine}[SIMD(:simd, 1, 32), Memory(:memory, 1, 2^55)]
    match_indices(physics_indices, machine_indices)
end

const layout_info_registers = Layout(
    Dict(
        IntValue(:intvalue, 1, 32) => SIMD(:simd, 1, 32),
        Index{Physics,ThreadTag}(:thread, 1, num_threads) => Thread(:thread, 1, num_threads),
        Index{Physics,WarpTag}(:warp, 1, num_warps) => Warp(:warp, 1, num_warps),
        Index{Physics,BlockTag}(:block, 1, num_blocks) => Block(:block, 1, num_blocks),
    ),
)

function make_xpose2048_kernel()
    # Prepare code generator
    emitter = Emitter(kernel_setup)

    # info: the kernel is starting
    apply!(emitter, :info => layout_info_registers, 1i32)
    store!(emitter, :info_memory => layout_info_memory, :info)

    load!(emitter, :scatter_indices => layout_scatter_indices_registers, :scatter_indices_memory => layout_scatter_indices_memory)

    loop!(emitter, Time(:time, T1, Tloop) => Loop(:time_loop, T1, Tloop)) do emitter
        load!(
            emitter,
            :E0 => layout_E_registers,
            :Ein_memory => layout_E_memory;
            align=16,
            postprocess=addr -> :(
                let
                    offset = $(shrinkmul(idiv(D, 4) * P * F, :Tinmin, T))
                    length = $(shrink(idiv(D, 4) * P * F * T))
                    mod($addr + offset, length)
                end
            ),
        )

        # dish0/simd0   <-> time0/reg2
        permute!(emitter, :E1, :E0, Register(:register, 4, 2), SIMD(:simd, 8, 2))
        # dish1/simd1   <-> time1/reg3
        permute!(emitter, :E2, :E1, Register(:register, 8, 2), SIMD(:simd, 16, 2))
        # dish4/thread0 <-> dish0/reg2
        permute!(emitter, :E3, :E2, Register(:register, 4, 2), Thread(:thread, 1, 2))
        # dish5/thread1 <-> dish1/reg3
        permute!(emitter, :E4, :E3, Register(:register, 8, 2), Thread(:thread, 2, 2))
        # dish6/thread2 <-> dish1/reg4
        permute!(emitter, :E5, :E4, Register(:register, 16, 2), Thread(:thread, 4, 2))

        # Avoid syncing on the first iteration
        if!(emitter, :(time_loop > 0i32)) do emitter
            sync_threads!(emitter)
            return nothing
        end

        layout_E5 = emitter.environment[:E5]
        isdish(kv::Pair) = ((k, v) = kv; k isa Dish)
        filter_dishes(l::Layout) = Layout(filter(isdish, l.dict))
        @assert filter_dishes(layout_scatter_indices_registers) ⊆ filter_dishes(layout_E5)

        # store!(
        #     emitter, :E_shared => layout_E_shared, :E5; postprocess=addr -> quote
        #         let
        #             addr = $addr
        #             time = addr % 8i32
        #             src_dish = addr ÷ 8i32
        #             dst_dish = scatter_indices_memory[src_dish + 0x1]
        #             time + 8i32 * dst_dish
        #         end
        #     end
        # )
        unrolled_loop!(emitter, Dish(:dish, 4, 32) => UnrolledLoop(:register_loop, 1, 32)) do emitter
            select!(emitter, :E5reg, :E5, Register(:register, 1, 32) => UnrolledLoop(:register_loop, 1, 32))
            select!(
                emitter, :scatter_indices_reg, :scatter_indices, Register(:register, 1, 32) => UnrolledLoop(:register_loop, 1, 32)
            )
            store!(emitter, :E_shared => layout_E_shared, :E5reg; postprocess=addr -> quote
                let
                    addr = $addr
                    time = addr % 8i32
                    dish = scatter_indices_reg
                    time + 8i32 * dish
                end
            end)
            return nothing
        end

        sync_threads!(emitter)

        load!(emitter, :E6 => layout_E5, :E_shared => layout_E_shared)

        # Swap in the opposite order from above
        # dish6/thread2 <-> dish1/reg4
        permute!(emitter, :E7, :E6, Register(:register, 16, 2), Thread(:thread, 4, 2))
        # dish5/thread1 <-> dish1/reg3
        permute!(emitter, :E8, :E7, Register(:register, 8, 2), Thread(:thread, 2, 2))
        # dish4/thread0 <-> dish0/reg2
        permute!(emitter, :E9, :E8, Register(:register, 4, 2), Thread(:thread, 1, 2))
        # dish1/simd1   <-> time1/reg3
        permute!(emitter, :E10, :E9, Register(:register, 8, 2), SIMD(:simd, 16, 2))
        # dish0/simd0   <-> time0/reg2
        permute!(emitter, :E11, :E10, Register(:register, 4, 2), SIMD(:simd, 8, 2))

        layout_E11 = emitter.environment[:E11]
        @assert layout_E11 == layout_E_registers

        store!(
            emitter,
            :E_memory => layout_E_memory,
            :E11;
            align=16,
            postprocess=addr -> :(
                let
                    offset = $(shrinkmul(idiv(D, 4) * P * F, :Tmin, T))
                    length = $(shrink(idiv(D, 4) * P * F * T))
                    mod($addr + offset, length)
                end
            ),
        )

        return nothing
    end

    # info: there were no errors
    apply!(emitter, :info => layout_info_registers, 0i32)
    store!(emitter, :info_memory => layout_info_memory, :info)

    # Emit code
    stmts = clean_code(
        quote
            @fastmath @inbounds begin
                $(emitter.init_statements...)
                $(emitter.statements...)
            end
        end,
    )

    return stmts
end

println("[Creating xpose2048 kernel...]")
const xpose2048_stmts = make_xpose2048_kernel()
println("[Done creating xpose2048 kernel]")

@eval function xpose2048_kernel(
    Tinmin::Int32, Tinmax::Int32, Tmin::Int32, Tmax::Int32, Ein_memory, E_memory, scatter_indices_memory, info_memory
)
    E_shared = @cuDynamicSharedMem(Int4x8, idiv(shmem_bytes, sizeof(Int8x4)))
    $xpose2048_stmts
    return nothing
end

function main(; compile_only::Bool=false, output_kernel::Bool=false)
    if !compile_only
        println("CHORD 2048 transpose kernel")
    end

    if output_kernel
        open("output-$card/xpose2048_$setup.jl", "w") do fh
            return println(fh, xpose2048_stmts)
        end
    end

    if !compile_only
        println("Compiling kernel...")
    end

    num_threads = kernel_setup.num_threads
    num_warps = kernel_setup.num_warps
    num_blocks = kernel_setup.num_blocks
    num_blocks_per_sm = kernel_setup.num_blocks_per_sm
    shmem_bytes = kernel_setup.shmem_bytes
    @assert num_warps * num_blocks_per_sm ≤ 32 # (???)
    @assert shmem_bytes ≤ 100 * 1024 # NVIDIA A10/A40 have 100 kB shared memory
    kernel = @cuda launch = false minthreads = num_threads * num_warps blocks_per_sm = num_blocks_per_sm xpose2048_kernel(
        Int32(0),
        Int32(0),
        Int32(0),
        Int32(0),
        CUDA.zeros(Int4x8, 0),
        CUDA.zeros(Int4x8, 0),
        CUDA.zeros(Int32, 0),
        CUDA.zeros(Int32, 0),
    )
    attributes(kernel.fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem_bytes

    if compile_only
        return nothing
    end

    if output_kernel
        ptx = read("output-$card/xpose2048_$setup.ptx", String)
        ptx = replace(ptx, r".extern .func gpu_([^;]*);"s => s".func gpu_\1.noreturn\n{\n\ttrap;\n}")
        open("output-$card/xpose2048_$setup.ptx", "w") do fh
            return write(fh, ptx)
        end
        kernel_symbol = match(r"\s\.globl\s+(\S+)"m, ptx).captures[1]
        open("output-$card/xpose2048_$setup.yaml", "w") do fh
            return print(
                fh,
                """
        --- !<tag:chord-observatory.ca/x-engine/kernel-description-1.0.0>
        kernel-description:
          name: "xpose2048"
          description: "transpose kernel"
          design-parameters:
            number-of-complex-components: $C
            number-of-dishes: $D
            number-of-frequencies: $F
            number-of-polarizations: $P
            number-of-timesamples: $T
          compile-parameters:
            minthreads: $(num_threads * num_warps)
            blocks_per_sm: $num_blocks_per_sm
          call-parameters:
            threads: [$num_threads, $num_warps]
            blocks: [$num_blocks]
            shmem_bytes: $shmem_bytes
          kernel-symbol: "$kernel_symbol"
          kernel-arguments:
            - name: "Tinmin"
              intent: in
              type: Int32
            - name: "Tinmax"
              intent: in
              type: Int32
            - name: "Tmin"
              intent: in
              type: Int32
            - name: "Tmax"
              intent: in
              type: Int32
            - name: "Ein"
              intent: in
              type: Int4
              indices: [C, D, P, F, T]
              shape: [$C, $D, $P, $F, $T]
              strides: [1, $C, $(C*D), $(C*D*P), $(C*D*P*F)]
            - name: "E"
              intent: out
              type: Int4
              indices: [C, D, P, F, T]
              shape: [$C, $D, $P, $F, $T]
              strides: [1, $C, $(C*D), $(C*D*P), $(C*D*P*F)]
            - name: "scatter_indices"
              intent: in
              type: Int32
              indices: [D, P]
              shape: [$D, $P]
              strides: [1, $D]
            - name: "info"
              intent: out
              type: Int32
              indices: [thread, warp, block]
              shapes: [$num_threads, $num_warps, $num_blocks]
              strides: [1, $num_threads, $(num_threads*num_warps)]
        ...
        """,
            )
        end
        cxx = read("kernels/xpose2048_template.cxx", String)
        cxx = Mustache.render(
            cxx,
            Dict(
                "kernel_name" => "Transpose2048_$setup",
                "kernel_design_parameters" => [
                    Dict("type" => "int", "name" => "cuda_number_of_complex_components", "value" => "$C"),
                    Dict("type" => "int", "name" => "cuda_number_of_dishes", "value" => "$D"),
                    Dict("type" => "int", "name" => "cuda_number_of_frequencies", "value" => "$F"),
                    Dict("type" => "int", "name" => "cuda_number_of_polarizations", "value" => "$P"),
                    Dict("type" => "int", "name" => "cuda_max_number_of_timesamples", "value" => "$T"),
                    Dict("type" => "int", "name" => "cuda_granularity_number_of_timesamples", "value" => "$T1"),
                ],
                "minthreads" => num_threads * num_warps,
                "num_blocks_per_sm" => num_blocks_per_sm,
                "num_threads" => num_threads,
                "num_warps" => num_warps,
                "num_blocks" => num_blocks,
                "shmem_bytes" => shmem_bytes,
                "kernel_symbol" => kernel_symbol,
                "kernel_arguments" => [
                    Dict(
                        "name" => "Tinmin",
                        "kotekan_name" => "Tinmin",
                        "type" => "int32",
                        "isoutput" => false,
                        "hasbuffer" => false,
                        "isscalar" => true,
                    ),
                    Dict(
                        "name" => "Tinmax",
                        "kotekan_name" => "Tinmax",
                        "type" => "int32",
                        "isoutput" => false,
                        "hasbuffer" => false,
                        "isscalar" => true,
                    ),
                    Dict(
                        "name" => "Tmin",
                        "kotekan_name" => "Tmin",
                        "type" => "int32",
                        "isoutput" => false,
                        "hasbuffer" => false,
                        "isscalar" => true,
                    ),
                    Dict(
                        "name" => "Tmax",
                        "kotekan_name" => "Tmax",
                        "type" => "int32",
                        "isoutput" => false,
                        "hasbuffer" => false,
                        "isscalar" => true,
                    ),
                    Dict(
                        "name" => "Ein",
                        "kotekan_name" => "gpu_mem_input_voltage",
                        "type" => "int4p4chime",
                        "axes" => [
                            Dict("label" => "D", "length" => D),
                            Dict("label" => "P", "length" => P),
                            Dict("label" => "F", "length" => F),
                            Dict("label" => "T", "length" => T),
                        ],
                        "isoutput" => false,
                        "hasbuffer" => true,
                    ),
                    Dict(
                        "name" => "E",
                        "kotekan_name" => "gpu_mem_output_voltage",
                        "type" => "int4p4chime",
                        "axes" => [
                            Dict("label" => "D", "length" => D),
                            Dict("label" => "P", "length" => P),
                            Dict("label" => "F", "length" => F),
                            Dict("label" => "T", "length" => T),
                        ],
                        "isoutput" => true,
                        "hasbuffer" => true,
                    ),
                    Dict(
                        "name" => "scatter_indices",
                        "kotekan_name" => "gpu_mem_scatter_indices",
                        "type" => "int32",
                        "axes" => [Dict("label" => "D", "length" => D), Dict("label" => "P", "length" => P)],
                        "isoutput" => false,
                        "hasbuffer" => true,
                    ),
                    Dict(
                        "name" => "info",
                        "kotekan_name" => "gpu_mem_info",
                        "type" => "int32",
                        "axes" => [
                            Dict("label" => "thread", "length" => num_threads),
                            Dict("label" => "warp", "length" => num_warps),
                            Dict("label" => "block", "length" => num_blocks),
                        ],
                        "isoutput" => true,
                        "hasbuffer" => false,
                    ),
                ],
            ),
        )
        write("output-$card/xpose2048_$setup.cxx", cxx)
    end

    println("Done.")
    return nothing
end

if CUDA.functional()
    # Output kernel
    open("output-$card/xpose2048_$setup.ptx", "w") do fh
        redirect_stdout(fh) do
            @device_code_ptx main(; compile_only=true)
        end
    end
    open("output-$card/xpose2048_$setup.sass", "w") do fh
        redirect_stdout(fh) do
            @device_code_sass main(; compile_only=true)
        end
    end
    # This call needs to happen after generating PTX code since it
    # modifies the generated PTX code
    main(; output_kernel=true)

    # # Run test
    # main(; run_selftest=true)

    # # Run benchmark
    # main(; nruns=10000)

    # # Regular run, also for profiling
    # main()
end
