# CHORD transpose kernel

using CUDA
using CUDASIMDTypes
using IndexSpaces
using Mustache

const card = "A40"

if CUDA.functional()
    println("[Choosing CUDA device...]")
    CUDA.device!(0)
    println(name(device()))
    @assert name(device()) == "NVIDIA $card"
end

idiv(i::Integer, j::Integer) = (@assert iszero(i % j); i ÷ j)

@enum CHORDTag CplxTag DishTag FreqTag PolrTag TimeTag ThreadTag WarpTag BlockTag

const Cplx = Index{Physics,CplxTag}
const Dish = Index{Physics,DishTag}
const Freq = Index{Physics,FreqTag}
const Polr = Index{Physics,PolrTag}
const Time = Index{Physics,TimeTag}

# Setup

# const setup = :chord
setup::Symbol

@static if setup ≡ :chord

    # Full CHORD
    const C = 2
    const T = 2048   #TODO 32768
    const D = 512
    const P = 2
    const F = 16

elseif setup ≡ :hirax

    # Full CHORD
    const C = 2
    const T = 2048   #TODO 32768
    const D = 256
    const P = 2
    const F = 16

elseif setup ≡ :pathfinder

    # CHORD pathfinder
    const C = 2
    const T = 2048   #TODO 32768
    const D = 64
    const P = 2
    const F = 128

else
    @assert false
end

const Dshort = 8
const Tshort = 16

@assert D % Dshort == 0
@assert T % Tshort == 0

const W = 16
const B = F

const Wp = 2                    # warps for polarizations
const Wt = idiv(W, Wp)          # warps for times

const Lt = idiv(idiv(T, Wt), Tshort) # loop iterations for times

const num_simd_bits = 32
const num_threads = 32
const num_warps = W
const num_blocks = B
const num_blocks_per_sm = 1

const cacheline_length = 32     # counting in UInt32
const shmem_bytes = 0

const kernel_setup = KernelSetup(num_threads, num_warps, num_blocks, num_blocks_per_sm, shmem_bytes)

function ilog2(i::Integer)
    j = round(Int, log2(i))
    @assert 2^j == i
    return j
end

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

const layout_Ein_memory = let
    physics_indices = Index{Physics}[
        IntValue(:intvalue, 1, 4),
        Cplx(:cplx, 1, C),
        Dish(:dish, 1, Dshort),
        Time(:time, 1, Tshort),
        Dish(:dish, Dshort, idiv(D, Dshort)),
        Polr(:polr, 1, P),
        Freq(:freq, 1, F),
        Time(:time, Tshort, idiv(T, Tshort)),
    ]
    machine_indices = Index{Machine}[SIMD(:simd, 1, num_simd_bits), Memory(:memory, 1, 2^55)]
    match_indices(physics_indices, machine_indices)
end

const layout_Eout_memory = let
    physics_indices = Index{Physics}[
        IntValue(:intvalue, 1, 4), Cplx(:cplx, 1, C), Dish(:dish, 1, D), Polr(:polr, 1, P), Freq(:freq, 1, F), Time(:time, 1, T)
    ]
    machine_indices = Index{Machine}[SIMD(:simd, 1, num_simd_bits), Memory(:memory, 1, 2^55)]
    match_indices(physics_indices, machine_indices)
end

const layout_Ein_registers = let
    physics_indices = Index{Physics}[
        IntValue(:intvalue, 1, 4),
        Cplx(:cplx, 1, C),
        Dish(:dish, 1, Dshort),
        Time(:time, 1, Tshort),
        Dish(:dish, Dshort, idiv(D, Dshort)),
        Time(:time, Tshort, idiv(idiv(idiv(T, Wt), Lt), Tshort)),
    ]
    machine_indices = Index{Machine}[
        SIMD(:simd, 1, num_simd_bits),
        Register(:register, 1, 4),
        Thread(:thread, 2, idiv(num_threads, 2)),
        Thread(:thread, 1, 2),
        Register(:register, 4, Tshort),
    ]
    dict = Dict{Index{Physics},Index{Machine}}(
        Polr(:polr, 1, P) => Warp(:warp, 1, Wp),
        Time(:Time, idiv(idiv(T, Wt), Lt), Lt) => Loop(:loop, 1, Lt),
        Time(:Time, idiv(T, Wt), Wt) => Warp(:warp, Wp, Wt),
        Freq(:freq, 1, F) => Block(:block, 1, B),
    )
    match_indices(physics_indices, machine_indices, dict)
end

const layout_Eout_registers = let
    physics_indices = Index{Physics}[
        IntValue(:intvalue, 1, 4), Cplx(:cplx, 1, C), Dish(:dish, 1, D), Time(:time, 1, idiv(idiv(T, Wt), Lt))
    ]
    machine_indices = Index{Machine}[
        SIMD(:simd, 1, num_simd_bits),
        Register(:register, 1, 4),
        Thread(:thread, 1, num_threads),
        Register(:register, 4, Tshort),
    ]
    dict = Dict{Index{Physics},Index{Machine}}(
        Polr(:polr, 1, P) => Warp(:warp, 1, Wp),
        Time(:Time, idiv(idiv(T, Wt), Lt), Lt) => Loop(:loop, 1, Lt),
        Time(:Time, idiv(T, Wt), Wt) => Warp(:warp, Wp, Wt),
        Freq(:freq, 1, F) => Block(:block, 1, B),
    )
    match_indices(physics_indices, machine_indices, dict)
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

function make_xpose_kernel()
    # Prepare code generator
    emitter = Emitter(kernel_setup)

    # info: the kernel is starting
    apply!(emitter, :info => layout_info_registers, 1i32)
    store!(emitter, :info_memory => layout_info_memory, :info)

    loop!(emitter, Time(:time, Tshort, Lt) => Loop(:loop, 1, Lt)) do emitter
        load!(emitter, :Ein => layout_Ein_registers, :Ein_memory => layout_Ein_memory; align=16)

        if setup === :chord

            # Input layout:
            # dish:4/2   => register:1/2
            # dish:8/2   => thread:16/2
            # dish:16/2  => thread:1/2
            # dish:32/16 => register:4/16
            # time:1/2   => register:2/2
            # time:2/8   => thread:2/8

            # dish:4/2  is already correct.
            # dish:16/2 is already correct.

            # Move dish:8/2 to register:2/2
            permute!(emitter, :E2, :Ein, Dish(:dish, 8, 2), Register(:register, 2, 2))

            # Move dish:32/2 to thread:2/2
            # Move dish:64/2 to thread:4/2
            # Move dish:128/2 to thread:8/2
            # Move dish:256/2 to thread:16/2
            permute!(emitter, :E3, :E2, Dish(:dish, 32, 2), Thread(:thread, 2, 2))
            permute!(emitter, :E4, :E3, Dish(:dish, 64, 2), Thread(:thread, 4, 2))
            permute!(emitter, :E5, :E4, Dish(:dish, 128, 2), Thread(:thread, 8, 2))
            permute!(emitter, :Eout, :E5, Dish(:dish, 256, 2), Thread(:thread, 16, 2))

            # Output layout:
            # dish:4/4   => register:1/4
            # dish:16/32 => thread:1/32
            # time:1/16  => register:4/16

        elseif setup === :hirax

            # Input layout:
            # dish:4/2  => register:1/2
            # dish:8/2  => thread:16/2
            # dish:16/2 => thread:1/2
            # dish:32/8 => register:4/8
            # time:1/2  => register:2/2
            # time:2/8  => thread:2/8

            # dish:4/2  is already correct.
            # dish:16/2 is already correct.

            # Move dish:8/2 to register:2/2
            permute!(emitter, :E2, :Ein, Dish(:dish, 8, 2), Register(:register, 2, 2))

            # Move dish:32/2 to thread:2/2
            # Move dish:64/2 to thread:4/2
            # Move dish:128/2 to thread:8/2
            permute!(emitter, :E3, :E2, Dish(:dish, 32, 2), Thread(:thread, 2, 2))
            permute!(emitter, :E4, :E3, Dish(:dish, 64, 2), Thread(:thread, 4, 2))
            permute!(emitter, :Eout, :E4, Dish(:dish, 128, 2), Thread(:thread, 8, 2))

            # Output layout:
            # dish:4/4   => register:1/4
            # dish:16/16 => thread:1/16
            # time:1/2   => thread:16/2
            # time:2/8   => register:4/8

        elseif setup === :pathfinder

            # Input layout:
            # dish:4/2  => Machine.RegisterTag.register:1/2
            # dish:8/2  => Machine.ThreadTag.thread:16/2
            # dish:16/2 => Machine.ThreadTag.thread:1/2
            # dish:32/2 => Machine.RegisterTag.register:4/2
            # time:1/2  => Machine.RegisterTag.register:2/2
            # time:2/8  => Machine.ThreadTag.thread:2/8

            # dish:4/2  is already correct.
            # dish:16/2 is already correct.

            # Move dish:8/2 to register:2/2
            permute!(emitter, :E2, :Ein, Dish(:dish, 8, 2), Register(:register, 2, 2))

            # Move dish:32/2 to thread:2/2
            # Move dish:64/2 to thread:4/2
            permute!(emitter, :Eout, :E2, Dish(:dish, 32, 2), Thread(:thread, 2, 2))

            # Output layout:
            # dish:4/4  => Machine.RegisterTag.register:1/4
            # dish:16/4 => Machine.ThreadTag.thread:1/4
            # time:1/8  => Machine.ThreadTag.thread:4/8
            # time:8/2  => Machine.RegisterTag.register:4/2

        else
            @assert false
        end

        store!(emitter, :Eout_memory => layout_Eout_memory, :Eout; align=16)

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

println("[Creating xpose kernel...]")
const xpose_stmts = make_xpose_kernel()
println("[Done creating xpose kernel]")

@eval function xpose_kernel(Ein_memory, Eout_memory, info_memory)
    $xpose_stmts
    return nothing
end

function main(; compile_only::Bool=false, output_kernel::Bool=false)
    if !compile_only
        println("CHORD transpose kernel")
    end

    if output_kernel
        open("output-$card/xpose_$setup.jl", "w") do fh
            return println(fh, xpose_stmts)
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
    kernel = @cuda launch = false minthreads = num_threads * num_warps blocks_per_sm = num_blocks_per_sm xpose_kernel(
        CUDA.zeros(Int4x8, 0), CUDA.zeros(Int4x8, 0), CUDA.zeros(Int32, 0)
    )
    attributes(kernel.fun)[CUDA.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem_bytes

    if compile_only
        return nothing
    end

    if output_kernel
        ptx = read("output-$card/xpose_$setup.ptx", String)
        ptx = replace(ptx, r".extern .func gpu_([^;]*);"s => s".func gpu_\1.noreturn\n{\n\ttrap;\n}")
        open("output-$card/xpose_$setup.ptx", "w") do fh
            return write(fh, ptx)
        end
        kernel_symbol = match(r"\s\.globl\s+(\S+)"m, ptx).captures[1]
        open("output-$card/xpose_$setup.yaml", "w") do fh
            return print(
                fh,
                """
        --- !<tag:chord-observatory.ca/x-engine/kernel-description-1.0.0>
        kernel-description:
          name: "xpose"
          description: "transpose kernel"
          design-parameters:
            inner-number-of-dishes: $Dshort
            inner-number-of-timesamples: $Tshort
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
            - name: "Ein"
              intent: in
              type: Int4
              indices: [C, Dshort, Tshort, D, P, F, T]
              shape: [$C, $Dshort, $Tshort, $(idiv(D, Dshort)), $P, $F, $(idiv(T, Tshort))]
              strides: [1, $C, $(C * Dshort), $(C * Dshort * Tshort), $(C * Dshort * Tshort * idiv(D, Dshort)), $(C * Dshort * Tshort * idiv(D, Dshort) * P), $(C * Dshort * Tshort * idiv(D, Dshort) * P * F)]
            - name: "E"
              intent: in
              type: Int4
              indices: [C, D, P, F, T]
              shape: [$C, $D, $P, $F, $T]
              strides: [1, $C, $(C*D), $(C*D*P), $(C*D*P*F)]
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
        cxx = read("kernels/xpose_template.cxx", String)
        cxx = Mustache.render(
            cxx,
            Dict(
                "kernel_name" => "TransposeKernel_$setup",
                "kernel_design_parameters" => [
                    Dict("type" => "int", "name" => "cuda_number_of_complex_components", "value" => "$C"),
                    Dict("type" => "int", "name" => "cuda_number_of_dishes", "value" => "$D"),
                    Dict("type" => "int", "name" => "cuda_number_of_frequencies", "value" => "$F"),
                    Dict("type" => "int", "name" => "cuda_number_of_polarizations", "value" => "$P"),
                    Dict("type" => "int", "name" => "cuda_number_of_timesamples", "value" => "$T"),
                    Dict("type" => "int", "name" => "cuda_inner_number_of_dishes", "value" => "$Dshort"),
                    Dict("type" => "int", "name" => "cuda_inner_number_of_timesamples", "value" => "$Tshort"),
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
                        "name" => "Ein",
                        "kotekan_name" => "gpu_mem_voltage",
                        "type" => "int4p4",
                        "axes" => [
                            Dict("label" => "Dshort", "length" => Dshort),
                            Dict("label" => "Tshort", "length" => Tshort),
                            Dict("label" => "D", "length" => idiv(D, Dshort)),
                            Dict("label" => "P", "length" => P),
                            Dict("label" => "F", "length" => F),
                            Dict("label" => "T", "length" => idiv(T, Tshort)),
                        ],
                        "isoutput" => false,
                        "hasbuffer" => true,
                    ),
                    Dict(
                        "name" => "E",
                        "kotekan_name" => "gpu_mem_voltage",
                        "type" => "int4p4",
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
        write("output-$card/xpose_$setup.cxx", cxx)
    end

    println("Done.")
    return nothing
end

if CUDA.functional()
    # Output kernel
    open("output-$card/xpose_$setup.ptx", "w") do fh
        redirect_stdout(fh) do
            @device_code_ptx main(; compile_only=true)
        end
    end
    open("output-$card/xpose_$setup.sass", "w") do fh
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
