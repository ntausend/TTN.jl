function _dims_to_n_layer_quaternary(dims::NTuple{D, Int}) where D
    n_layer = 0
    try
        n_layer = Int64(sum(log.(4,dims)))
    catch
        throw(DimensionsException(dims))
    end
    return n_layer
end
#struct QuaternaryNetwork{D, S<:IndexSpace, I<:Sector} <: AbstractNetwork{D, S, I}
struct QuaternaryNetwork{L<:SimpleLattice} <: AbstractNetwork{L}
    lattices::Vector{L}
end

function QuaternaryNetwork(dimensions::NTuple{D,Int}, nd::Type{<:AbstractNode}; kwargs...) where{D}
    n_layer = _dims_to_n_layer_quaternary(dimensions)
    lat_vec = Vector{SimpleLattice{D}}(undef, n_layer + 1)

    dimensionsc = vcat(dimensions...)
    # first dimension must be largest, second second largest etc..
    # this is required due to our pairing
    if !(sort(dimensionsc) == reverse(dimensionsc))
        msg = "Only Lattices with first dimension being largest, second being second largest etc are suppported for QuaternaryNetworks. Dimensions: $dimensions"   
        throw(NotSupportedException(msg))
    end
    lat_vec[1] = SimpleLattice(dimensions, nd; kwargs...)

    vnd_type = nodetype(lat_vec[1])
    for jj in 2:n_layer+1
        dimensionsc = Int64.(dimensionsc ./ 2)
        
        lat = SimpleLattice(Tuple(dimensionsc), vnd_type)
        lat_vec[jj] = lat
        
        # pairing direction of the next layer
    end
    
    #return QuaternaryNetwork{D, spacetype(vnd_type), sectortype(vnd_type)}(lat_vec)
    return QuaternaryNetwork{typeof(lat_vec[1])}(lat_vec)
end
QuaternaryNetwork(dimensions::NTuple; kwargs...) = QuaternaryNetwork(dimensions, TrivialNode; kwargs...)

# creation from indices, may be fused with above function?
function QuaternaryNetwork(dims::NTuple{D, Int}, indices::Vector{<:Index}) where{D}
    @assert prod(dims) == length(indices)
    n_layer = _dims_to_n_layer_quaternary(dims)
    lat_vec = Vector{SimpleLattice{D}}(undef, n_layer + 1)

    dimensionsc = vcat(dims...)
    # first dimension must be largest, second second largest etc..
    # this is required due to our pairing
    if !(sort(dimensionsc) == reverse(dimensionsc))
        msg = "Only Lattices with first dimension being largest, second being second largest etc are suppported for QuaternaryNetworks. Dimensions: $dimensions"   
        throw(NotSupportedException(msg))
    end
    lat_vec[1] = SimpleLattice(dims, indices)
    vnd_type = nodetype(lat_vec[1])

    for jj in 2:n_layer+1
        dimensionsc = Int64.(dimensionsc ./ 2)

        lat = SimpleLattice(Tuple(dimensionsc), vnd_type)
        lat_vec[jj] = lat
    end
    
    return QuaternaryNetwork{typeof(lat_vec[1])}(lat_vec)
end
# creation from ITensorNode with type specifier
function QuaternaryNetwork(dims::NTuple{D, Int}, nd::Type{<:ITensorNode}, type::AbstractString; kwargs...) where{D}
    indices = siteinds(type, prod(dims); kwargs...)
    return QuaternaryNetwork(dims, indices)
end
QuaternaryNetwork(dims, type; kwargs...) = QuaternaryNetwork(dims, ITensorNode, type; kwargs...)


# number of child nodes of position. For quaternary networks this is a constant of 4
number_of_child_nodes(::QuaternaryNetwork, ::Tuple{Int,Int}) = 4
# exact formular for this kind of network
number_of_tensors(net::QuaternaryNetwork) = sum([4^i for i in 0:(number_of_layers(net)-1)])


# const QuaternaryChainNetwork{L<:SimpleLattice{1}} = QuaternaryNetwork{L}
# 
# function QuaternaryChainNetwork(number_of_layers::Int, nd::Type{<:AbstractNode}; kwargs...)
#     tensors_per_layer = [2^(number_of_layers - jj) for jj in 0:number_of_layers]
#     phys_lat = Chain(tensors_per_layer[1], nd; kwargs...)
#     
#     nvd_type = nodetype(phys_lat)
#     lat_vec = map(nn -> Chain(nn, nvd_type), tensors_per_layer)
#     lat_vec[1] = phys_lat
#     
#     #return QuaternaryChainNetwork{spacetype(nvd_type), sectortype(nvd_type)}(lat_vec)
#     return QuaternaryChainNetwork{typeof(lat_vec[1])}(lat_vec)
# end
# 
# QuaternaryChainNetwork(number_of_layers::Int; kwargs...) = QuaternaryChainNetwork(number_of_layers, TrivialNode; kwargs...)
# 
# function QuaternaryChainNetwork(number_of_sites::Tuple{Int}, nd::Type{<:AbstractNode}; kwargs...)
#     n_layers = _dims_to_n_layer_quaternary(Tuple(number_of_sites))
#     return QuaternaryChainNetwork(n_layers, nd; kwargs...)
# end
# QuaternaryChainNetwork(number_of_sites::Tuple{Int}; kwargs...) = QuaternaryChainNetwork(number_of_sites, TrivialNode; kwargs...)
# 
# function QuaternaryChainNetwork(indices::Vector{<:Index})
#     number_of_layers =  _dims_to_n_layer_quaternary((length(indices),))
#     tensors_per_layer = [2^(number_of_layers - jj) for jj in 0:number_of_layers]
#     phys_lat = Chain(indices)
#     nvd_type = nodetype(phys_lat)
#     lat_vec = map(nn -> Chain(nn,nvd_type), tensors_per_layer) 
#     lat_vec[1] = phys_lat
#     return QuaternaryChainNetwork{typeof(lat_vec[1])}(lat_vec)
# end
# 
# function QuaternaryChainNetwork(number_of_layers::Int, nd::Type{<:ITensorNode}, type::AbstractString; kwargs...)
#     indices = siteinds(type, 2^number_of_layers; kwargs...)
#     return QuaternaryChainNetwork(indices)
# end
# function QuaternaryChainNetwork(number_of_sites::Tuple{Int}, nd::Type{<:ITensorNode}, type::AbstractString; kwargs...)
#     n_layers = _dims_to_n_layer_quaternary(Tuple(number_of_sites))
#     return QuaternaryChainNetwork(n_layers, nd, type)
# end
# 
# """
# ```julia
#     ChainQuaternaryChainNetwork(number_of_sites::Int, type::AbstractString; kwargs...)
# ```
# 
# Creates a quaternary tree network based on a one dimensional chain of length `number_of_sites`.
# 
# # Arguments
# 
# ---
# 
# - `number_of_sites`: Length of the chain
# - `type`: Definition of the local Hilbert-space
# 
# # Keywords
# 
# ---
# 
# The keywords `kwargs` are passed to the `siteinds` function and defines the properties of the local Hilbert-space
# 
# """
# QuaternaryChainNetwork(number_of_sites::Int, type::AbstractString; kwargs...) = QuaternaryChainNetwork(number_of_sites, ITensorNode, type; kwargs...) 
# 
# 
# function QuaternaryRectangularNetwork(number_of_layers::Int, nd::Type{<:AbstractNode}, args...; kwargs...)
#     
#     # calculate the physical dimensions and use the fall back function
#     # number of layers is always representitive as n_l = 2*n + r
#     # the number of sites in x direction is 2^((2n + r + 1) / 2) since
#     # we start with a pairing in x direction. In case r == 1 this gives
#     # an extra doubling in x. On the other hand, the number of sites in y direction
#     # are given by 2^n. If r == 0 the both numbers conincidend and we have a square lattice.
# 
#     n_x = 2^(div(number_of_layers + 1, 2))
#     n_y = 2^(div(number_of_layers, 2))
# 
#     return QuaternaryNetwork((n_x, n_y), nd, args...; kwargs...)
# end
# QuaternaryRectangularNetwork(number_of_layers::Int; kwargs...) = QuaternaryRectangularNetwork(number_of_layers, TrivialNode; kwargs...)
# 
# """
# ```julia
#     QuaternaryRectangularNetwork(number_of_layers::Int, type::AbstractString; kwargs...)
# ```
# 
# Creates a quaternary tree network based on a two dimensional rectangle with `number_of_layers` layers. The physical Hilbert-space is given by `type`.
# 
# # Arguments
# 
# ---
# 
# - `number_of_layers`: Number of layers in the network.
# - `type`: Definition of the local Hilbert-space
# 
# # Keywords
# 
# ---
# 
# The keywords `kwargs` are passed to the `siteinds` function and defines the properties of the local Hilbert-space
# 
# """
# QuaternaryRectangularNetwork(number_of_layers::Int, type::AbstractString, kwargs...) = QuaternaryRectangularNetwork(number_of_layers, ITensorNode, type; kwargs...)
# 
# #QuaternaryRectangularNetwork(dims::Tuple{Int,Int}, nd::Type{<:AbstractNode}, args...; kwargs...) = QuaternaryNetwork(dims, nd; kwargs...)
# """
# ```julia
#     QuaternaryRectangularNetwork(dims::Tuple{Int,Int}, type::AbstractString; kwargs...)
# ```
# 
# Creates a quaternary tree network based on a two dimensional rectangle with dimensions given by `dims`. The physical Hilbert-space is given by `type`.
# 
# # Arguments
# 
# ---
# 
# - `dims`: A two dimensional tuple defining the dimensionality of the rectangle
# - `type`: Definition of the local Hilbert-space
# 
# # Keywords
# 
# ---
# 
# The keywords `kwargs` are passed to the `siteinds` function and defines the properties of the local Hilbert-space
# 
# """
# QuaternaryRectangularNetwork(dims::Tuple{Int,Int}, type::AbstractString; kwargs...) = QuaternaryNetwork(dims, ITensorNode, type; kwargs...)
# QuaternaryRectangularNetwork(dims::Tuple{Int,Int}; kwargs...) = QuaternaryNetwork(dims, TrivialNode; kwargs...)
# 
# QuaternaryRectangularNetwork(dims::Tuple{Int,Int}, indices::Vector{<:Index}) = QuaternaryNetwork(dims, indices)
# QuaternaryRectangularNetwork(indices::Matrix{<:Index}) = QuaternaryNetwork(size(indices), vec(indices))


function parent_node(net::QuaternaryNetwork, pos::Tuple{Int, Int})
    check_valid_position(net, pos)
    pos[1] == number_of_layers(net) && (return nothing)

    D = dimensionality(net) #dimensionality_reduced(net, pos[1])

    # # unroll the linear index
    pos_vec = vcat(_coordinate_simple_lattice(pos[2], dimensions(net, pos[1]))...)
    pos_vec = div.(pos_vec .+ 1, 2)

    return (pos[1] + 1, _linear_ind_simple_lattice(Tuple(pos_vec), dimensions(net, pos[1] + 1)))
end

function child_nodes(net::QuaternaryNetwork, pos::Tuple{Int, Int})
    check_valid_position(net, pos)
    
    pos[1] == 0 && (return nothing)
    D = dimensionality(net) #dimensionality_reduced(net, pos[1] - 1)
    @assert D == 2

    pair_dir = mod(pos[1] - 1, dimensionality(net)) + 1
    if dimensions(net, pos[1]-1)[pair_dir] == 1
      pair_dir = mod(pair_dir+1, 2) 
    end
    
    # getting the coordinates inside the current layer
    pos_vec = vcat(_coordinate_simple_lattice(pos[2], dimensions(net, pos[1]))...)

    # getting the dimensions of the lower layer.
    dims_ll = dimensions(net, pos[1] - 1)

    p1 = 2 .* pos_vec .- 1
    p2 = p1 .+ [1,0]
    p3 = p1 .+ [0,1]
    p4 = p1 .+ [1,1]

    return [(pos[1] - 1, _linear_ind_simple_lattice(Tuple(p1), dims_ll)), 
            (pos[1] - 1, _linear_ind_simple_lattice(Tuple(p2), dims_ll)),
            (pos[1] - 1, _linear_ind_simple_lattice(Tuple(p3), dims_ll)),
            (pos[1] - 1, _linear_ind_simple_lattice(Tuple(p4), dims_ll))]
end

function index_of_child(net::QuaternaryNetwork, pos_child::Tuple{Int,Int})
    pos_vec = _coordinate_simple_lattice(pos_child[2], dimensions(net, pos_child[1]))
    pos_vec = mod1.(pos_vec,2)

    return first(pos_vec) + 2*(last(pos_vec)-1)
end

# function adjacency_matrix(net::QuaternaryNetwork, l::Int)
#     l == number_of_layers(net) && return nothing
# 	n_this = number_of_tensors(net, l)
# 	n_next = number_of_tensors(net, l+1)
#     I  = zeros(Int64, n_this)
#     J  = collect(1:n_this)
# 
#     for jj in 1:n_this
#         parent_idx = parent_node(net, (l,jj))
#         I[jj] = parent_idx[2]
#     end
#     return sparse(I,J, repeat([1], n_this), n_next, n_this)
# end
# 
# function adjacency_matrix(net::QuaternaryNetwork{L}, l::Int64) where{L<:SimpleLattice{1}}
#     l == number_of_layers(net) && return nothing
# 	n_this = number_of_tensors(net, l)
# 	n_next = number_of_tensors(net, l+1)
# 	pos_this = collect(1:n_this)
# 	I = repeat(collect(1:n_next), 2)
# 	J = vcat(pos_this[1:2:end], pos_this[2:2:end])
# 	return sparse(I,J,repeat([1], n_this), n_next, n_this)
# end
# 
# # find a good overload function for this one
# #=
# function internal_index_of_legs(net::QuaternaryChainNetwork, pos::Tuple{Int,Int})
#     n_layers = number_of_layers(net)
#     number_of_childs_prev_layers = 2^(n_layers - pos[1] + 1)
#     n_shift_1 = 2^(pos[2] - 1) + number_of_childs_prev_layers 
#     n_shift_2 = 2^(n_layers - pos[2] + 2)
# 	return [1 + n_shift_1, 2 + n_shift_1, n_shift_2 + pos[2]]
# end
# =#
