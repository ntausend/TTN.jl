function _dims_to_n_layer_binary(dims::NTuple{D, Int}) where D
    n_layer = 0
    try
        n_layer = sum(Int64.(log2.(dims)))
    catch
        throw(DimensionsException(dims))
    end
    return n_layer
end
#struct BinaryNetwork{D, S<:IndexSpace, I<:Sector} <: AbstractNetwork{D, S, I}
struct BinaryNetwork{L<:SimpleLattice} <: AbstractNetwork{L}
    lattices::Vector{L}
end

function BinaryNetwork(dimensions::NTuple{D,Int}, nd::Type{<:AbstractNode}; kwargs...) where{D}
    n_layer = _dims_to_n_layer_binary(dimensions)
    lat_vec = Vector{SimpleLattice{D}}(undef, n_layer + 1)

    dimensionsc = vcat(dimensions...)
    # first dimension must be largest, second second largest etc..
    # this is required due to our pairing
    if !(sort(dimensionsc) == reverse(dimensionsc))
        msg = "Only Lattices with first dimension being largest, second being second largest etc are suppported for BinaryNetworks. Dimensions: $dimensions"   
        throw(NotSupportedException(msg))
    end
    lat_vec[1] = SimpleLattice(dimensions, nd; kwargs...)

    vnd_type = nodetype(lat_vec[1])
    for jj in 2:n_layer+1
        # this is not working...
        D_actual = D #- sum(dimensionsc[2:end][dimensionsc[2:end] .== 1])
        pair_dir  = mod1(jj-1, D_actual)
        dimensionsc[pair_dir] = div(dimensionsc[pair_dir],2)
        
        #dimensionsc[dimensionsc.==0] .= 1
        lat = SimpleLattice(Tuple(dimensionsc), vnd_type)
        lat_vec[jj] = lat
        
        # pairing direction of the next layer
    end
    
    #return BinaryNetwork{D, spacetype(vnd_type), sectortype(vnd_type)}(lat_vec)
    return BinaryNetwork{typeof(lat_vec[1])}(lat_vec)
end
BinaryNetwork(dimensions::NTuple; kwargs...) = BinaryNetwork(dimensions, TrivialNode; kwargs...)

# creation from indices, may be fused with above function?
function BinaryNetwork(dims::NTuple{D, Int}, indices::Vector{<:Index}) where{D}
    @assert prod(dims) == length(indices)
    n_layer = _dims_to_n_layer_binary(dims)
    lat_vec = Vector{SimpleLattice{D}}(undef, n_layer + 1)

    dimensionsc = vcat(dims...)
    # first dimension must be largest, second second largest etc..
    # this is required due to our pairing
    if !(sort(dimensionsc) == reverse(dimensionsc))
        msg = "Only Lattices with first dimension being largest, second being second largest etc are suppported for BinaryNetworks. Dimensions: $dimensions"   
        throw(NotSupportedException(msg))
    end
    lat_vec[1] = SimpleLattice(dims, indices)
    vnd_type = nodetype(lat_vec[1])

    for jj in 2:n_layer+1
        pair_dir  = mod1(jj-1, D)
        if dimensionsc[pair_dir] == 1
          pair_dir = mod(pair_dir+1, 2) 
        end
        dimensionsc[pair_dir] = div(dimensionsc[pair_dir],2)
        dimensionsc[dimensionsc.==0] .= 1

        lat = SimpleLattice(Tuple(dimensionsc), vnd_type)
        lat_vec[jj] = lat
        # pairing direction of the next layer
    end
    
    #return BinaryNetwork{D, spacetype(vnd_type), sectortype(vnd_type)}(lat_vec)
    return BinaryNetwork{typeof(lat_vec[1])}(lat_vec)
end
# creation from ITensorNode with type specifier
function BinaryNetwork(dims::NTuple{D, Int}, nd::Type{<:ITensorNode}, type::AbstractString; kwargs...) where{D}
    indices = siteinds(type, prod(dims); kwargs...)
    return BinaryNetwork(dims, indices)
end
BinaryNetwork(dims, type; kwargs...) = BinaryNetwork(dims, ITensorNode, type; kwargs...)


# number of child nodes of position. For binary networks this is a constant of 2
number_of_child_nodes(::BinaryNetwork, ::Tuple{Int,Int}) = 2
# exact formular for this kind of network
number_of_tensors(net::BinaryNetwork) = 2^(number_of_layers(net)) - 1


#const BinaryChainNetwork{S<:IndexSpace, I<:Sector} = BinaryNetwork{1, S, I}
const BinaryChainNetwork{L<:SimpleLattice{1}} = BinaryNetwork{L}

function BinaryChainNetwork(number_of_layers::Int, nd::Type{<:AbstractNode}; kwargs...)
    tensors_per_layer = [2^(number_of_layers - jj) for jj in 0:number_of_layers]
    phys_lat = Chain(tensors_per_layer[1], nd; kwargs...)
    
    nvd_type = nodetype(phys_lat)
    lat_vec = map(nn -> Chain(nn, nvd_type), tensors_per_layer)
    lat_vec[1] = phys_lat
    
    #return BinaryChainNetwork{spacetype(nvd_type), sectortype(nvd_type)}(lat_vec)
    return BinaryChainNetwork{typeof(lat_vec[1])}(lat_vec)
end

BinaryChainNetwork(number_of_layers::Int; kwargs...) = BinaryChainNetwork(number_of_layers, TrivialNode; kwargs...)

function BinaryChainNetwork(number_of_sites::Tuple{Int}, nd::Type{<:AbstractNode}; kwargs...)
    n_layers = _dims_to_n_layer_binary(Tuple(number_of_sites))
    return BinaryChainNetwork(n_layers, nd; kwargs...)
end
BinaryChainNetwork(number_of_sites::Tuple{Int}; kwargs...) = BinaryChainNetwork(number_of_sites, TrivialNode; kwargs...)

function BinaryChainNetwork(indices::Vector{<:Index})
    number_of_layers =  _dims_to_n_layer_binary((length(indices),))
    tensors_per_layer = [2^(number_of_layers - jj) for jj in 0:number_of_layers]
    phys_lat = Chain(indices)
    nvd_type = nodetype(phys_lat)
    lat_vec = map(nn -> Chain(nn,nvd_type), tensors_per_layer) 
    lat_vec[1] = phys_lat
    return BinaryChainNetwork{typeof(lat_vec[1])}(lat_vec)
end

function BinaryChainNetwork(number_of_layers::Int, nd::Type{<:ITensorNode}, type::AbstractString; kwargs...)
    indices = siteinds(type, 2^number_of_layers; kwargs...)
    return BinaryChainNetwork(indices)
end
function BinaryChainNetwork(number_of_sites::Tuple{Int}, nd::Type{<:ITensorNode}, type::AbstractString; kwargs...)
    n_layers = _dims_to_n_layer_binary(Tuple(number_of_sites))
    return BinaryChainNetwork(n_layers, nd, type)
end

"""
```julia
    ChainBinaryChainNetwork(number_of_sites::Int, type::AbstractString; kwargs...)
```

Creates a binary tree network based on a one dimensional chain of length `number_of_sites`.

# Arguments

---

- `number_of_sites`: Length of the chain
- `type`: Definition of the local Hilbert-space

# Keywords

---

The keywords `kwargs` are passed to the `siteinds` function and defines the properties of the local Hilbert-space

"""
BinaryChainNetwork(number_of_sites::Int, type::AbstractString; kwargs...) = BinaryChainNetwork(number_of_sites, ITensorNode, type; kwargs...) 


function BinaryRectangularNetwork(number_of_layers::Int, nd::Type{<:AbstractNode}, args...; kwargs...)
    
    # calculate the physical dimensions and use the fall back function
    # number of layers is always representitive as n_l = 2*n + r
    # the number of sites in x direction is 2^((2n + r + 1) / 2) since
    # we start with a pairing in x direction. In case r == 1 this gives
    # an extra doubling in x. On the other hand, the number of sites in y direction
    # are given by 2^n. If r == 0 the both numbers conincidend and we have a square lattice.

    n_x = 2^(div(number_of_layers + 1, 2))
    n_y = 2^(div(number_of_layers, 2))

    return BinaryNetwork((n_x, n_y), nd, args...; kwargs...)
end
BinaryRectangularNetwork(number_of_layers::Int; kwargs...) = BinaryRectangularNetwork(number_of_layers, TrivialNode; kwargs...)

"""
```julia
    BinaryRectangularNetwork(number_of_layers::Int, type::AbstractString; kwargs...)
```

Creates a binary tree network based on a two dimensional rectangle with `number_of_layers` layers. The physical Hilbert-space is given by `type`.

# Arguments

---

- `number_of_layers`: Number of layers in the network.
- `type`: Definition of the local Hilbert-space

# Keywords

---

The keywords `kwargs` are passed to the `siteinds` function and defines the properties of the local Hilbert-space

"""
BinaryRectangularNetwork(number_of_layers::Int, type::AbstractString, kwargs...) = BinaryRectangularNetwork(number_of_layers, ITensorNode, type; kwargs...)

#BinaryRectangularNetwork(dims::Tuple{Int,Int}, nd::Type{<:AbstractNode}, args...; kwargs...) = BinaryNetwork(dims, nd; kwargs...)
"""
```julia
    BinaryRectangularNetwork(dims::Tuple{Int,Int}, type::AbstractString; kwargs...)
```

Creates a binary tree network based on a two dimensional rectangle with dimensions given by `dims`. The physical Hilbert-space is given by `type`.

# Arguments

---

- `dims`: A two dimensional tuple defining the dimensionality of the rectangle
- `type`: Definition of the local Hilbert-space

# Keywords

---

The keywords `kwargs` are passed to the `siteinds` function and defines the properties of the local Hilbert-space

"""
BinaryRectangularNetwork(dims::Tuple{Int,Int}, type::AbstractString; kwargs...) = BinaryNetwork(dims, ITensorNode, type; kwargs...)
BinaryRectangularNetwork(dims::Tuple{Int,Int}; kwargs...) = BinaryNetwork(dims, TrivialNode; kwargs...)

BinaryRectangularNetwork(dims::Tuple{Int,Int}, indices::Vector{<:Index}) = BinaryNetwork(dims, indices)
BinaryRectangularNetwork(indices::Matrix{<:Index}) = BinaryNetwork(size(indices), vec(indices))


function parent_node(net::BinaryNetwork, pos::Tuple{Int, Int})
    check_valid_position(net, pos)
    pos[1] == number_of_layers(net) && (return nothing)
    # we need to be carefull if the layer is completely paired
    # not working
    D = dimensionality(net)#dimensionality_reduced(net, pos[1])
    # check if paring is along x or y direction in the next step
    # even layers are paired along the x direction, odd layers along the y direction
    pair_dir = mod(pos[1], dimensionality(net)) + 1
    if dimensions(net, pos[1])[pair_dir] == 1
      pair_dir = mod(pair_dir+1, 2) 
    end
    # unroll the linear index
    pos_vec = vcat(_coordinate_simple_lattice(pos[2], dimensions(net, pos[1]))...)

    # the coordinate of the pairing dimension is given by either
    #           p_j = 2n - 1 or 2n
    # both coordinates are maped to the parent coordinate p̃ = n
    # which are then converted back to the linear index of the next layer.
    pos_vec[pair_dir] = div(pos_vec[pair_dir] + 1, 2)

    return (pos[1] + 1, _linear_ind_simple_lattice(Tuple(pos_vec), dimensions(net, pos[1] + 1)))
end

function parent_node(net::BinaryNetwork{L}, pos::Tuple{Int, Int}) where {L<:SimpleLattice{1}}
    check_valid_position(net, pos)
    pos[1] == number_of_layers(net) && (return nothing)

    return(pos[1] + 1, div(pos[2] + 1,2))
end


function child_nodes(net::BinaryNetwork, pos::Tuple{Int, Int})
    check_valid_position(net, pos)
    
    pos[1] == 0 && (return nothing)

    # we need to be carefull if the layer is completely paired
    # not working
    D = dimensionality(net)#dimensionality_reduced(net, pos[1] - 1)

    # do the revert operation as for the parent nodes
    # pairing of this layer, given by the pairing direction of
    # the previous layer
    pair_dir = mod(pos[1] - 1, dimensionality(net)) + 1
    if dimensions(net, pos[1]-1)[pair_dir] == 1
      pair_dir = mod(pair_dir+1, 2) 
    end
    
    # getting the coordinates inside the current layer
    pos_vec = vcat(_coordinate_simple_lattice(pos[2], dimensions(net, pos[1]))...)

    # getting the dimensions of the lower layer.
    #dims_ll = pos[1] == 1 ? size(lattice(net)) : dimensions(net, pos[1] - 1)
    dims_ll = dimensions(net, pos[1] - 1)

    p1 = copy(pos_vec)
    p2 = copy(pos_vec)
    p1[pair_dir] = 2*p1[pair_dir] - 1 
    p2[pair_dir] = 2*p2[pair_dir]

    return [(pos[1] - 1, _linear_ind_simple_lattice(Tuple(p1), dims_ll)), 
            (pos[1] - 1, _linear_ind_simple_lattice(Tuple(p2), dims_ll))]
end

function child_nodes(net::BinaryNetwork{L}, pos::Tuple{Int, Int}) where {L<:SimpleLattice{1}}
    check_valid_position(net, pos)
    pos[1] == 0 && (return nothing)

    return [(pos[1] - 1, 2*pos[2] - 1), (pos[1] - 1, 2*pos[2])]
end

function index_of_child(net::BinaryNetwork, pos_child::Tuple{Int,Int})
    pair_dir = mod(pos_child[1], dimensionality(net)) + 1
    pos_vec = _coordinate_simple_lattice(pos_child[2], dimensions(net, pos_child[1]))
    return mod1(pos_vec[pair_dir], 2)
end

index_of_child(::BinaryNetwork{L}, pos_child::Tuple{Int,Int}) where{L<:SimpleLattice{1}} = mod1(pos_child[2], 2)


function adjacency_matrix(net::BinaryNetwork, l::Int)
    l == number_of_layers(net) && return nothing
	n_this = number_of_tensors(net, l)
	n_next = number_of_tensors(net, l+1)
    I  = zeros(Int64, n_this)
    J  = collect(1:n_this)

    for jj in 1:n_this
        parent_idx = parent_node(net, (l,jj))
        I[jj] = parent_idx[2]
    end
    return sparse(I,J, repeat([1], n_this), n_next, n_this)
end

function adjacency_matrix(net::BinaryNetwork{L}, l::Int64) where{L<:SimpleLattice{1}}
    l == number_of_layers(net) && return nothing
	n_this = number_of_tensors(net, l)
	n_next = number_of_tensors(net, l+1)
	pos_this = collect(1:n_this)
	I = repeat(collect(1:n_next), 2)
	J = vcat(pos_this[1:2:end], pos_this[2:2:end])
	return sparse(I,J,repeat([1], n_this), n_next, n_this)
end

# find a good overload function for this one
#=
function internal_index_of_legs(net::BinaryChainNetwork, pos::Tuple{Int,Int})
    n_layers = number_of_layers(net)
    number_of_childs_prev_layers = 2^(n_layers - pos[1] + 1)
    n_shift_1 = 2^(pos[2] - 1) + number_of_childs_prev_layers 
    n_shift_2 = 2^(n_layers - pos[2] + 2)
	return [1 + n_shift_1, 2 + n_shift_1, n_shift_2 + pos[2]]
end
=#
