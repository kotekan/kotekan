# This is a placeholder - some Julia tools want the "src" directory to exist and want
# a module with the same name as the package.  In the future we might actually build
# out this package to interface the Kotekan code to Julia.
module Kotekan

using ASDF2
using ArchGDAL
using DimensionalData
using CUDASIMDTypes
using GDAL
using MappedArrays

const AG = ArchGDAL

export c2t, i2c, i2t, t2c
c2t(x::Complex) = (real(x), imag(x))
i2c(x::Int4x2) = t2c(i2t(x))
i2t(x::Int4x2) = convert(NTuple{2,Int8}, x)
t2c(x::NTuple{2}) = Complex(x...)

export u4p42i8
u4p42i8(x::UInt8) = (((x >>> 0x0) & 0x0f) % Int8, ((x >>> 0x4) & 0x0f) % Int8)

export u162f16
u162f16(x::UInt16) = reinterpret(Float16, x)

export read_asdf
function read_asdf(filename::AbstractString, quantity::AbstractString, indexnames::Union{AbstractVector,Tuple})
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

export read_gdal
function read_gdal(filename::AbstractString)
    filename = expanduser(filename)

    # @info "using iteration $iter..."
    # @info "reading dataset \"$datasetname\"..."
    dataset = AG.open(
        filename, AG.OF_MULTIDIM_RASTER | AG.OF_READONLY | AG.OF_SHARED | AG.OF_VERBOSE_ERROR, nothing, nothing, nothing
    )
    root = AG.getrootgroup(dataset)

    # attrs = AG.getname.(AG.getattributes(root))
    # AG.getmdarraynames(root)

    name = AG.readattribute(root, "name")::AbstractString
    @info "found dataset \"$name\""
    type = AG.readattribute(root, "type")::AbstractString
    @info "found dataset type $type"

    mdarray = AG.openmdarray(root, name)

    dims = AG.getdimensions(mdarray)
    dimnames = Tuple(AG.getname.(dims))
    @info "found index names $dimnames"
    dimsizes = Tuple(AG.getsize.(dims))
    @info "found dataset size $dimsizes"

    data = AG.readmdarray(root, name)
    @assert size(data) == dimsizes

    # Convert type if necessary
    if type == "float16"
        @assert eltype(data) == UInt16
        @info "mapping to Float16..."
        data = reinterpret(Float16, data)
    elseif type == "int4p4chime"
        @assert eltype(data) == UInt8
        @info "mapping to Complex{Int8}..."
        data = mappedarray(i2c ∘ Int4x2, data)
    elseif type == "uint4p4"
        @assert eltype(data) == UInt8
        @info "mapping to Int8..."
        data = reinterpret(Int8, mappedarray(u4p42i8, data))
        # dims[1] is now wrong!
        dimsizes = Base.setindex(dimsizes, 2*dimsizes[1], 1)
    end
    if dimnames[begin] == "C"
        @info "mapping to Complex..."
        data = reinterpret(reshape, Complex{eltype(data)}, data)
        dims = dims[begin+1:end]
        dimnames = dimnames[begin+1:end]
        dimsizes = dimsizes[begin+1:end]
    end

    # Apply DimArray; do this last, it doesn't survive `mappedarray`
    data = DimArray(data, ntuple(d -> Dim{Symbol(dimnames[d])}(1:dimsizes[d]), length(dimnames)))

    return data::AbstractArray
end

end
