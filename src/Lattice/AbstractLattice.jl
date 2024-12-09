function _check_dimensions(dims::NTuple{D, Int}) where D
    try
        sum(Int64.(log2.(dims)))
    catch
        throw(DimensionsException(dims))
    end
end

abstract type AbstractLattice{D, S, I} end

dimensionality(::Type{<:AbstractLattice{D}}) where D = D
dimensionality(lat::AbstractLattice) = dimensionality(typeof(lat))

# extracting indices, only necessary for ITensors backend
function ITensors.siteinds(lat::AbstractLattice)
    is_physical(lat) || error("Site indices only meaningful for physical lattices")
    return map(hilbertspace, lat)
end



nodes(lat::AbstractLattice) = lat.lat
node(lat::AbstractLattice, p::Int) = nodes(lat)[p]

number_of_sites(lat::AbstractLattice) = length(lat.lat)

# convertions from linear to D-dim tuple and back, not implemented in generallity, functionallity has to be
# implemented for derived types
linear_ind(lat::AbstractLattice{D}, ::NTuple{D,Int}) where D = throw(NotImplemented(:linear_ind, typeof(lat)))
coordinate(lat::AbstractLattice{D}, ::Int) where {D} = throw(NotImplemented(:coordinate, typeof(lat)))

# check if lattice is made of physical nodes
is_physical(lat::AbstractLattice) = all(map(x-> x isa PhysicalNode, lat))

# pass through node copy method, assumes all nodes to be the same -> may change in the future?
nodetype(lat::AbstractLattice) = nodetype((node(lat,1)))

sectortype(::Type{<:AbstractLattice{D,S,I}}) where{D,S,I} = I
sectortype(lat::AbstractLattice) = sectortype(typeof(lat))
spacetype(::Type{<:AbstractLattice{D,S}}) where{D,S} = S
spacetype(lat::AbstractLattice) = spacetype(typeof(lat)) 

function coordinates(lat::AbstractLattice)
    coord = map(x-> coordinate(lat, x), eachindex(lat))
    return coord
end

# base functions overloading
Base.iterate(lat::AbstractLattice) = iterate(lat.lat)
Base.iterate(lat::AbstractLattice, state) = iterate(lat.lat, state)
Base.length(lat::AbstractLattice) = number_of_sites(lat)

Base.getindex(lat::AbstractLattice, jj::Int) = node(lat, jj)

function ==(::AbstractLattice, ::AbstractLattice)
    # not implemented so far for general lattices
    return false
end


Base.eachindex(la::AbstractLattice) = 1:number_of_sites(la)


# general lattice -> Maybe removed in the future?
struct GenericLattice{D, S, I} <: AbstractLattice{D, S, I}
    lat::Vector{Node{S,I}}
    dims::NTuple{D, Int}
end
Base.size(lat::GenericLattice) = lat.dims

# only for homogenous hilberstapces with trivial sectors currently
function CreateChain(number_of_sites::Int) 
    lat_vec = [Node(n, "$n", backend) for n in 1:number_of_sites]
    _check_dimensions((number_of_sites,))
    return GenericLattice{1, spacetype(lat_vec[1]), sectortype(lat_vec[1])}(lat_vec, (number_of_sites,))
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, lattice::AbstractLattice)
    g = create_group(parent, name)
    g["dims"] = collect(lattice.dims)
    for (num_node, node) in enumerate(lattice.lat)
        name_node = "node_"*string(num_node)
        write(g, name_node, node)
    end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{AbstractLattice})
    g = open_group(parent, name)
    dims = Tuple(read(g, "dims"))
    lat = map(1:prod(dims)) do i
        name_node = "node_$(i)"
        read(g, name_node, AbstractNode)
    end
    return SimpleLattice{length(dims),Index,Int64}(lat, dims)
end
