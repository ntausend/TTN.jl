
abstract type AbstractLattice{D, S<:IndexSpace, I<:Sector} end

nodes(la::AbstractLattice) = la.lat
node(la::AbstractLattice, p::Int) = nodes(la)[p]
#local_dim(la::AbstractLattice, p::Int) = dim(hilbertspace(node(la,p)))
#local_dims(la::AbstractLattice)   = [local_dim(la, jj) for jj in 1:number_of_sites(la)]
TensorKit.sectortype(::AbstractLattice{D,S,I}) where{D,S,I} = I
space(::AbstractLattice{D,S}) where{D,S} = S
dimensionality(::AbstractLattice{D}) where D = D

number_of_sites(la::AbstractLattice) = length(la.lat)

to_linear_ind(::AbstractLattice{D}, ::NTuple{D,Int}) where D = error("Tuple to linear Index not implemented for general lattices.")

to_coordinate(::AbstractLattice{D}, ::Int) where {D} = error("Linear Index to Tuple not implemented for general lattices.")

is_physical(lat::AbstractLattice) = all(map(x-> x isa PhysicalNode, lat))

parentNode(la::AbstractLattice, p::NTuple{D,Int}) where D = parentNode(la, to_linear_ind(la,p))

VirtualNode(la::AbstractLattice) = VirtualNode(node(la,1))

function coordinates(lat::AbstractLattice)
    map(x-> to_coordinate(lat, x), eachsite(lat))
end

# general lattice
struct Lattice{D, S<:IndexSpace, I<:Sector} <: AbstractLattice{D, S, I}
    lat::Vector{Node{S,I}}
    dims::NTuple{D, Int}
end

# only for homogenous hilberstapces with trivial sectors currently
function CreateBinaryChain(number_of_sites::Int; field::Type{<:EuclideanSpace} = ComplexSpace)
    lat_vec = [Node(n, "$n"; field = field) for n in 1:number_of_sites]
    n_layer = 0
    try
        n_layer = Int64(log2(number_of_sites))
    catch
        error("Number of Sites $number_of_sites is not compatible with a binary network of n 
              layers requireing number_of_sites = 2^n")
    end
    n_first_layer = 2^(n_layer-1)
    pos_phys = collect(1:number_of_sites)
    I = repeat(collect(1:n_first_layer), 2)
    J = vcat(pos_phys[1:2:end], pos_phys[2:2:end])
 
    return Lattice{1, space(lat_vec[1]), sectortype(lat_vec[1])}(lat_vec, (number_of_sites,))
end


Base.iterate(la::AbstractLattice) = iterate(la.lat)
Base.iterate(la::AbstractLattice, state) = iterate(la.lat, state)
Base.length(la::AbstractLattice) = number_of_sites(la)

import Base: ==
function ==(::AbstractLattice, ::AbstractLattice)
    # not implemented so far for general lattices
    return false
end

eachsite(la::AbstractLattice) = 1:number_of_sites(la)