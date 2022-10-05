function _to_linear_ind(p::NTuple{D,Int}, dims::NTuple{D,Int}) where{D}
    res = mapreduce(+,enumerate(p[2:end]), init = p[1]) do (jj, pp)
        (pp-1)*prod(dims[1:jj])
    end
    return res
end

function _to_coordinate(p::Int, dims::NTuple{D,Int}) where {D}
    p_vec = Vector{Int64}(undef, D)
    p_r = p
    for jj in D:-1:2
        modifier = prod(dims[1:jj-1])
        p_vec[jj] = div(p_r - 1, modifier) + 1
        p_r = mod1(p_r, modifier)
    end
    p_vec[1] = p_r
    return Tuple(p_vec)
end

struct BinaryLattice{D, S<:IndexSpace, I<:Sector} <: AbstractLattice{D,S,I}
    lat::Vector{AbstractNode{S, I}}
    dims::NTuple{D,Int}
end

function _dims_to_n_layer(dims::NTuple{D, Int}) where D
    n_layer = 0
    try
        n_layer = sum(Int64.(log2.(dims)))
    catch
        n_sites = prod(dims)
        s_err = "Number of Sites $(n_sites) is not compatible with a binary network of dimension $D"
        s_err *= " with n layers requireing number_of_sites = 2^(n_$(D))"
        error(s_err) 
    end
    return n_layer
end

"""
    BinaryLattice(dims::NTuple{D, Int}, nodetype::Function; kwargs...) where D

Construction of a general Binary Lattice of dimension `D`. This Lattice serves as a basis for
the Binary networks, where every layer is represented as a binary lattice.
Every dimension should be multiple of 2 in order to be compatible with a Binary Network
(and thus the name `BinaryLattice`).

## Arguments:
- `dims::NTuple{D,Int}`: A `D` dimensional Tuple defining the dimensions of the lattice.
   every entry should be a multiple of 2, i.e. `dims[j] = 2^(n_j)` should be fulfilled for
   every 0≤ j≤ D.
- `nd`: Valid Node type, will be used for factory as `nd(l, n)` where `l` is the linear position
   and `n` is a description string
"""
function BinaryLattice(dims::NTuple{D, Int}, nd::Type{<:AbstractNode}; kwargs...) where {D}
    # checking if dims are in the correct layout
    n_layer = _dims_to_n_layer(dims)

    prod_it = Iterators.product(UnitRange.(1, dims)...)
    
    nd_names = map(prod_it) do p_ind
        mapreduce(p -> " $p", *, p_ind[2:end], init=string(p_ind[1]))
    end
    lin_inds = map(p -> _to_linear_ind(p, dims), prod_it)
    lat_vec = map(zip(lin_inds, nd_names)) do (l, n)
        nd(l, n; kwargs...)
    end
    lat_vec = vec(lat_vec)
    return BinaryLattice{length(dims), space(lat_vec[1]), sectortype(lat_vec[1])}(lat_vec, dims)
end

# Fast factory of a lattice with trivial nodes with hilbertspace dimension local_dim
function BinaryLattice(dims::NTuple{D, Int}, dim::Int; field = ComplexSpace) where D
    return BinaryLattice(dims, TrivialNode; dim = dim, field = field)
end

#const BinaryChain = BinaryLattice{1}
BinaryChain(n_sites::Int, dim::Int; field = ComplexSpace) = BinaryLattice((n_sites,), dim; field = field)
BinaryChain(n_sites::Int, nd::Type{<:AbstractNode}; kwargs...) = BinaryLattice((n_sites,), nd; kwargs...)


#const BinaryRectangle = BinaryLattice{2}
BinaryRectangle(n_x::Int, n_y::Int, dim::Int; field = ComplexSpace) = BinaryLattice((n_x, n_y), dim; field = field)
BinaryRectangle(dims::Tuple{Int, Int}, dim::Int; field = ComplexSpace) = BinaryLattice(dims, dim; field = field)
BinarySquare(n_lin::Int, dim::Int; field = ComplexSpace) = BinaryRectangle(n_lin, n_lin, dim; field = field)
BinaryRectangle(n_x::Int, n_y::Int, nd::Type{<:AbstractNode}; kwargs...) = BinaryLattice((n_x, n_y), nd; kwargs...)
BinaryRectangle(dims::Tuple{Int, Int}, nd::Type{<:AbstractNode}; kwargs...) = BinaryLattice(dims, nd; kwargs...)
BinarySquare(n_lin::Int, nd::Type{<:AbstractNode}; kwargs...) = BinaryRectangle(n_lin, n_lin, nd; kwargs...)


Base.size(lat::BinaryLattice) = lat.dims
Base.size(lat::BinaryLattice, d::Integer) = size(lat)[d]

function to_linear_ind(lat::BinaryLattice{D}, p::NTuple{D,Int}) where D
    return _to_linear_ind(p, size(lat))
end

function to_coordinate(lat::BinaryLattice{D}, p::Int) where {D}
    return _to_coordinate(p, size(lat))
end


#=
# for the BinaryLattice types we assume paring of the sites along the x axis for
# the first layer. With this one has (x is the fast changing index):
#       (2x - 1, y) ∧ (2x, y) -> (l=1, x,y)
#       (2x-1, y) -> 2x-1 + (y-1)*dim_x 
#       (2x, y)   -> 2x   + (y-1)*dim_x 
# and since dim_x is a multiple of 2, the resulting linear_index are adjacent
# if and only if the coordiantes are adjacent along the x direction.
# This may change in future if we allow for odd number of layers in the x direction
# in this case, one has to unroll the linear index to the coordiante form, calculated
# the parent node in coordainate form and casting this back to the linear index:
#       xx, yy = to_coordinate(lat, p_lin)
#       px = (xx + 1) ÷ 2 # or equivalent for odd numbered sites... has to be deceided
#       pos_new = to_linear_ind(lat)((px + (yy-1) * dim_x÷2))

function parentNode(lat::BinaryLattice, p::Int)
    @warn "This function is depricated and might not work..."
    @assert 0 < p ≤ number_of_sites(lat)
    return (1, div((p+1),2))
end



# again simple through the choice of paring along the x direction
function adjacencyMatrix(lat::BinaryLattice{D}) where {D}
    @warn "This function is depricated and might not work..."
    n_sites = number_of_sites(lat)
    n_layer = _dims_to_n_layer(dims)

    n_first_layer = 2^(n_layer-1)

    pos_phys = collect(1:n_sites)
    I = repeat(collect(1:n_first_layer), 2)
    J = vcat(pos_phys[1:2:end], pos_phys[2:2:end])
 
	return sparse(I,J,repeat([1], n_sites), n_first_layer, n_sites)
end
=#

import Base: ==
function ==(lat1::BinaryLattice{D1}, lat2::BinaryLattice{D2}) where{D1, D2}
    D1 == D2 || return false    
    all(size(lat1) .== size(lat2)) || return false
    is_physical(lat1) == is_physical(lat2) || return false

    isph = is_physical(lat1)

    are_equal = mapreduce(*, zip(lat1, lat2), init = true) do (nd1, nd2)
        if(isph)
            return hilbertspace(nd1) == hilbertspace(nd2)
        else
            return sectortype(nd1) == sectortype(nd2) && space(nd1) == space(nd2)
        end
    end
    return are_equal
end


import Base: show
function show(io::IO, lat::BinaryLattice{D}) where{D}
    println(io, "Binary Lattice of dimension $D, with dimensions: $(size(lat)):\n")
    if( D == 1)
        s = "\t"
        lengths = Vector{Int64}(undef, length(lat))
        for (jj,nd) in enumerate(lat)
            #desc = description(nd)
            if nd isa PhysicalNode
                desc = string(hilbertspace(nd))
            else
                desc = string(sectortype(nd))
            end
            if (jj == 1)
                s_coord = "|$(desc)|"
            else
                s_coord = "-|$(desc)|"
            end
            s *= s_coord
            lengths[jj] = length(s_coord)
        end
        print(io, s)
        println(io, "")
        
        s = "\t"
        for jj in 1:length(lat)
            l_half = lengths[jj]÷2
            l_res  = lengths[jj] - l_half
            s_lower = repeat(" ", l_half) * "|" * repeat(" ", l_res-1)
            s *= s_lower
        end
        print(io,s)
        println(io,"")
    elseif (D == 2 )
        lengths = Matrix{Int64}(undef, size(lat))
        for yy in 1:size(lat,2)
            s = "\t"
            for xx in 1:size(lat,1)
                #desc = description(node(lat, to_linear_ind(lat, (xx,yy))))
                nd = node(lat, to_linear_ind(lat, (xx,yy)))
                if nd isa PhysicalNode
                    desc = string(hilbertspace(nd))
                else
                    desc = string(sectortype(nd))
                end
                if xx == 1
                    s_coord = "|$(desc)|"
                else
                    s_coord = "-|$(desc)|"
                end
                s *= s_coord
                lengths[xx,yy] = length(s_coord)
            end
            print(io, s)
            println(io, "")
            yy == size(lat,2) && break
            s = "\t"
            for jj in 1:size(lat,1)
                l_half = lengths[jj,yy]÷2
                l_res  = lengths[jj,yy] - l_half
                s_lower = repeat(" ", l_half) * "|" * repeat(" ", l_res-1)
                s *= s_lower
            end
            print(io,s)
            println(io,"")
        end
    else
        for nd in lat
            print(io, "\t")
            println(io, nd)
        end
    end
    # just implement a show for one and two dimensional cases..
    #D>2 && println(io, lat)
end