include("./Node.jl")

abstract type AbstractLattice{D} end
node(la::AbstractLattice, p::Int64) = la.lat[p]
local_dim(la::AbstractLattice)   =  la.local_dim
adjacencyMatrix(la::AbstractLattice) = la.adjMat
number_of_sites(la::AbstractLattice) = length(la.lat)
to_linear_ind(la::AbstractLattice)   = la.to_linear_ind
to_coordinate(la::AbstractLattice)   = la.to_coordinate
local_hilbertspace(la::AbstractLattice) = la.local_hilbertspace

#abstract type BinaryLattice{D} <: AbstractLattice{D} end


function parentNode(la::AbstractLattice, p::Int64)
    adjMat = adjacencyMatrix(la)


	idx_parent = findall(!iszero, adjMat[:,p])
	length(idx_parent) > 1 && error("Number of parents not exactly 1. Illdefined Tree Tensor Network")
	return (1, idx_parent[1])
end

parentNode(la::AbstractLattice{D}, p::NTuple{D,Int64}) where{D} = parentNode(la, to_linear_ind(la)(p))

#function n_childs(la::AbstractLattice{D})

# general lattice
struct Lattice{D} <: AbstractLattice{D}
    lat::Vector{Node}
    adjMat::SparseMatrixCSC{Int64,Int64}
    local_dim::Int64
    local_hilbertspace
    to_linear_ind
    to_coordinate
end

function CreateBinaryChain(number_of_sites::Int64, local_dim::Int64; field = ComplexSpace)
    lat_vec = [Node(n, "$n") for n in 1:number_of_sites]
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

    return Lattice{1}(lat_vec, adj_mat, local_dim, field(local_dim), x -> x[1])
end


Base.iterate(la::AbstractLattice) = iterate(la.lat)
Base.iterate(la::AbstractLattice, state) = iterate(la.lat, state)
Base.length(la::AbstractLattice) = length(la.lat)

import Base: ==
function ==(la1::AbstractLattice, la2::AbstractLattice)
    # not implemented so far for general lattices
    return false
end