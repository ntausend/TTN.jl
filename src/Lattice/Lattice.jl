include("./Node.jl")

abstract type AbstractLattice{D} end

node(la::AbstractLattice, p::Int) = la.lat[p]
local_dim(la::AbstractLattice, p::Int) = dim(hilbertspace(node(la,p)))
local_dims(la::AbstractLattice)   = [local_dim(la, jj) for jj in 1:number_of_sites(la)]
dimensionality(::AbstractLattice{D}) where D = D

adjacencyMatrix(la::AbstractLattice) = la.adjMat
number_of_sites(la::AbstractLattice) = length(la.lat)

function to_linear_ind(la::AbstractLattice, p::NTuple{D,Int}) where D
    @assert dimensionality(la) == D
    dims = size(la)
    res = mapreduce(+,enumerate(p[2:end]), init = p[1]) do (jj, pp)
        (pp-1)*prod(dims[1:jj])
    end
    return res
end

function to_coordinate(la::AbstractLattice{D}, p::Int) where {D}
    p = Vector{Int64}(undef, D)
    dim = size(la)
    error("Need implementing general formular...")
end

Base.size(la::AbstractLattice) = la.dims
Base.size(la::AbstractLattice, d::Integer) = size(la)[d]


function parentNode(la::AbstractLattice, p::Int)
    adjMat = adjacencyMatrix(la)

	idx_parent = findall(!iszero, adjMat[:,p])
	length(idx_parent) > 1 && error("Number of parents not exactly 1. Ill-defined Tree Tensor Network")
	return (1, idx_parent[1])
end

parentNode(la::AbstractLattice, p::NTuple{D,Int}) where D = parentNode(la, to_linear_ind(la,p))

# general lattice
struct Lattice{D} <: AbstractLattice{D}
    lat::Vector{Node}
    adjMat::SparseMatrixCSC{Int,Int}
    dims::NTuple{D, Int}
end

# only for homogenous hilberstapces with trivial sectors currently
function CreateBinaryChain(number_of_sites::Int, local_dim::Int; field = ComplexSpace)
    lat_vec = [Node(n, field(local_dim), "$n") for n in 1:number_of_sites]
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
 
	adj_mat = sparse(I,J,repeat([1], number_of_sites), n_first_layer, number_of_sites)

    return Lattice{1}(lat_vec, adj_mat, (number_of_sites,))
end


Base.iterate(la::AbstractLattice) = iterate(la.lat)
Base.iterate(la::AbstractLattice, state) = iterate(la.lat, state)
Base.length(la::AbstractLattice) = number_of_sites(la)

import Base: ==
function ==(::AbstractLattice, ::AbstractLattice)
    # not implemented so far for general lattices
    return false
end