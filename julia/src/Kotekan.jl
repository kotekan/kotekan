# This is a placeholder - some Julia tools want the "src" directory to exist and want
# a module with the same name as the package.  In the future we might actually build
# out this package to interface the Kotekan code to Julia.
module Kotekan

using ASDF2
using CUDASIMDTypes
using MappedArrays

export c2t, i2c, i2t, t2c
c2t(x::Complex) = (real(x), imag(x))
i2c(x::Int4x2) = t2c(i2t(x))
i2t(x::Int4x2) = convert(NTuple{2,Int8}, x)
t2c(x::NTuple{2}) = Complex(x...)

export read_kotekan
function read_kotekan(filename::AbstractString, quantity::AbstractString, indexnames::Union{AbstractVector,Tuple})
    if !isempty(indexnames)
        indexnames::Union{AbstractVector{<:AbstractString},NTuple{N,<:AbstractString} where N}
    end

    # Open file
    file = ASDF2.load_file(filename)

    # Use first iteration
    iters = filter(k -> k isa Integer, keys(file.metadata))
    iter = sort(collect(iters))[begin]
    @info "using iteration $iter..."
    dataset = file.metadata[iter]

    # Read data
    datasetname = "host_$(quantity)_buffer"
    @info "reading dataset \"$datasetname\"..."
    data = dataset[datasetname][]
    @info "found dataset size $(size(data))"
    @info "found dataset type $(eltype(data))"

    # Convert type if necessary
    if eltype(data) == UInt8
        @info "mapping to Complex{Int8}..."
        data = mappedarray(i2c ∘ Int4x2, data)
    end

    # Permute indices
    data_indexnames = reverse(dataset["dim_names"])
    @info "found index names $data_indexnames"
    @assert length(data_indexnames) == length(indexnames)
    data_name2index = Dict(name => dir for (dir, name) in enumerate(data_indexnames))
    perm = [data_name2index[indexnames[dir]] for dir in 1:length(indexnames)]
    needperm = any(perm[dir] != dir for dir in 1:length(indexnames))
    if needperm
        @info "permuting indices via $perm..."
        data = PermutedDimsArray(data, perm)
    end

    return data::AbstractArray
end

end
