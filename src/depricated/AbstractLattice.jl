function _check_dimensions(dims::NTuple{D, Int}) where D
    try
        sum(Int64.(log2.(dims)))
    catch
        throw(DimensionsException(dims))
    end
end

abstract type AbstractLattice{D, S<:IndexSpace, I<:Sector} end

dimensionality(::Type{<:AbstractLattice{D}}) where D = D
dimensionality(lat::AbstractLattice) = dimensionality(typeof(lat))


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


TensorKit.sectortype(::Type{<:AbstractLattice{D,S,I}}) where{D,S,I} = I
TensorKit.sectortype(lat::AbstractLattice) = sectortype(typeof(lat))
TensorKit.spacetype(::Type{<:AbstractLattice{D,S}}) where{D,S} = S
TensorKit.spacetype(lat::AbstractLattice) = spacetype(typeof(lat)) 

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
struct Lattice{D, S<:IndexSpace, I<:Sector} <: AbstractLattice{D, S, I}
    lat::Vector{Node{S,I}}
    dims::NTuple{D, Int}
end
Base.size(lat::Lattice) = lat.dims

# only for homogenous hilberstapces with trivial sectors currently
function CreateChain(number_of_sites::Int; field::Type{<:EuclideanSpace} = ComplexSpace)
    lat_vec = [Node(n, "$n"; field = field) for n in 1:number_of_sites]
    _check_dimensions((number_of_sites,))
    return Lattice{1, spacetype(lat_vec[1]), sectortype(lat_vec[1])}(lat_vec, (number_of_sites,))
end