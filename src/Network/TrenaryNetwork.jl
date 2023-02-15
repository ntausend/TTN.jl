function _dims_to_n_layer_trenary(dims::NTuple{D, Int}) where{D}
    n_layer = 0
    try
        n_layer = sum(Int64.(log.(3, dims)))
    catch
        throw(DimensionsException(dims))
    end
    return n_layer
end


struct TrenaryNetwork{L<:SimpleLattice, B<:AbstractBackend} <: AbstractNetwork{L,B}
    lattices::Vector{L}
end


function TrenaryNetwork(dims::NTuple{D, Int}, indices::Vector{<:Index}) where{D}
    @assert prod(dims) == length(indices)
    
    n_layer = _dims_to_n_layer_trenary(dims)
    lat_vec = Vector{SimpleLattice{D}}(undef, n_layer + 1)
    
    dimensionsc = vcat(dims...)
    # first dimension must be largest, second second largest etc..
    # this is required due to our pairing
    if !(sort(dimensionsc) == reverse(dimensionsc))
        msg = "Only Lattices with first dimension beeing largest, second being second largest etc are suppported for Trenary Networks. Dimensions: $dimensions"   
        throw(NotSupportedException(msg))
    end
    
    lat_vec[1] = SimpleLattice(dims, indices)
    vnd_type = nodetype(lat_vec[1])
    
    for jj in 2:n_layer+1
        D_actual = D - sum(dimensionsc[2:end][dimensionsc[2:end] .== 1])
        pair_dir  = mod1(jj-1, D_actual)
        dimensionsc[pair_dir] = div(dimensionsc[pair_dir],3)
        #dimensionsc[dimensionsc.==0] .= 1

        lat = SimpleLattice(Tuple(dimensionsc), vnd_type)
        lat_vec[jj] = lat
        # pairing direction of the next layer
    end
    return TrenaryNetwork{typeof(lat_vec[1]), ITensorsBackend}(lat_vec)
end


# creation from ITensorNode with type specifier
function TrenaryNetwork(dims::NTuple{D, Int}, nd::Type{<:ITensorNode}, type::AbstractString; kwargs...) where{D}
    indices = siteinds(type, prod(dims); kwargs...)
    return TrenaryNetwork(dims, indices)
end


# number of child nodes of position. For binary networks this is a constant of 2
number_of_child_nodes(::TrenaryNetwork, ::Tuple{Int,Int}) = 3
# exact formular for this kind of network
# general formular for q, n_L = number of layers
# n_T = (q^n_L - 1)/(q - 1)
number_of_tensors(net::TrenaryNetwork) = Int64((3^number_of_layers(net) - 1)/2)

#const BinaryChainNetwork{S<:IndexSpace, I<:Sector} = BinaryNetwork{1, S, I}
const TrenaryChainNetwork{L<:SimpleLattice{1}, B} = TrenaryNetwork{L,B}
function TrenaryChainNetwork(number_of_layers::Int, nd::Type{<:AbstractNode}; kwargs...)
    tensors_per_layer = [3^(number_of_layers - jj) for jj in 0:number_of_layers]
    phys_lat = Chain(tensors_per_layer[1], nd; kwargs...)
    _backend = backend(phys_lat) 
    
    nvd_type = nodetype(phys_lat)
    lat_vec = map(nn -> Chain(nn, nvd_type), tensors_per_layer)
    lat_vec[1] = phys_lat
    
    #return BinaryChainNetwork{spacetype(nvd_type), sectortype(nvd_type)}(lat_vec)
    return TrenaryChainNetwork{typeof(lat_vec[1]), _backend}(lat_vec)
end
TrenaryChainNetwork(number_of_layers::Int; kwargs...) = TrenaryChainNetwork(number_of_layers, TrivialNode; kwargs...)

function TrenaryChainNetwork(number_of_sites::Tuple{Int}, nd::Type{<:AbstractNode}; kwargs...)
    n_layers = _dims_to_n_layer_trenary(Tuple(number_of_sites))
    return TrenaryChainNetwork(n_layers, nd; kwargs...)
end
TrenaryChainNetwork(number_of_sites::Tuple{Int}; kwargs...) = TrenaryChainNetwork(number_of_sites, TrivialNode; kwargs...)



function TrenaryChainNetwork(indices::Vector{<:Index})
    number_of_layers =  _dims_to_n_layer_trenary((length(indices),))
    tensors_per_layer = [3^(number_of_layers - jj) for jj in 0:number_of_layers]
    phys_lat = Chain(indices)
    nvd_type = nodetype(phys_lat)
    lat_vec = map(nn -> Chain(nn,nvd_type), tensors_per_layer) 
    lat_vec[1] = phys_lat
    return TrenaryChainNetwork{typeof(lat_vec[1]), backend(phys_lat)}(lat_vec)
end

function TrenaryChainNetwork(number_of_layers::Int, nd::Type{<:ITensorNode}, type::AbstractString; kwargs...)
    indices = siteinds(type, 3^number_of_layers; kwargs...)
    return TrenaryChainNetwork(indices)
end
function TrenaryChainNetwork(number_of_sites::Tuple{Int}, nd::Type{<:ITensorNode}, type::AbstractString; kwargs...)
    n_layers = _dims_to_n_layer_trenary(Tuple(number_of_sites))
    return TrenaryChainNetwork(n_layers, nd, type)
end


function TrenaryRectangularNetwork(number_of_layers::Int, nd::Type{<:AbstractNode}, args...; kwargs...)
    
    # calculate the physical dimensions and use the fall back function
    # number of layers is always representitive as n_l = 2*n + r
    # the number of sites in x direction is 2^((2n + r + 1) / 2) since
    # we start with a pairing in x direction. In case r == 1 this gives
    # an extra doubling in x. On the other hand, the number of sites in y direction
    # are given by 2^n. If r == 0 the both numbers conincidend and we have a square lattice.

    n_x = 3^(div(number_of_layers + 1, 3))
    n_y = 3^(div(number_of_layers, 3))

    return TrenaryNetwork((n_x, n_y), nd, args...; kwargs...)
end
TrenaryRectangularNetwork(number_of_layers::Int; kwargs...) = TrenaryRectangularNetwork(number_of_layers, TrivialNode; kwargs...)

TrenaryRectangularNetwork(dims::Tuple{Int,Int}, nd::Type{<:AbstractNode}, args...; kwargs...) = TrenaryNetwork(dims, nd; kwargs...)
TrenaryRectangularNetwork(dims::Tuple{Int,Int}; kwargs...) = TrenaryNetwork(dims, TrivialNode; kwargs...)

TrenaryRectangularNetwork(dims::Tuple{Int,Int}, indices::Vector{<:Index}) = TrenaryNetwork(dims, indices)
TrenaryRectangularNetwork(indices::Matrix{<:Index}) = TrenaryNetwork(size(indices), vec(indices))




function parent_node(net::TrenaryNetwork, pos::Tuple{Int, Int})
    check_valid_position(net, pos)
    pos[1] == number_of_layers(net) && (return nothing)
    # we need to be carefull if the layer is completely paired
    D = dimensionality_reduced(net, pos[1])
    # check if paring is along x or y direction in the next step
    # even layers are paired along the x direction, odd layers along the y direction
    pair_dir = mod(pos[1], D) + 1
    
    # unroll the linear index
    pos_vec = vcat(_coordinate_simple_lattice(pos[2], dimensions(net, pos[1]))...)

    # the coordinate of the pairing dimension is given by either
    #           p_j = 3n - 2, 3n - 1 or 3n
    # all coordinates are maped to the parent coordinate p̃ = n
    # which are then converted back to the linear index of the next layer.
    pos_vec[pair_dir] = div(pos_vec[pair_dir] + 2, 3)

    return (pos[1] + 1, _linear_ind_simple_lattice(Tuple(pos_vec), dimensions(net, pos[1] + 1)))
end


function parent_node(net::TrenaryNetwork{L}, pos::Tuple{Int, Int}) where {L<:SimpleLattice{1}}
    check_valid_position(net, pos)
    pos[1] == number_of_layers(net) && (return nothing)

    return(pos[1] + 1, div(pos[2] + 2,3))
end


function child_nodes(net::TrenaryNetwork, pos::Tuple{Int, Int})
    check_valid_position(net, pos)
    
    pos[1] == 0 && (return nothing)
    D = dimensionality_reduced(net, pos[1] - 1)
    # do the revert operation as for the parent nodes
    # pairing of this layer, given by the pairing direction of
    # the previous layer
    pair_dir = mod(pos[1] - 1, D) + 1
    
    # getting the coordinates inside the current layer
    pos_vec = vcat(_coordinate_simple_lattice(pos[2], dimensions(net, pos[1]))...)

    # getting the dimensions of the lower layer.
    #dims_ll = pos[1] == 1 ? size(lattice(net)) : dimensions(net, pos[1] - 1)
    dims_ll = dimensions(net, pos[1] - 1)

    p1 = copy(pos_vec)
    p2 = copy(pos_vec)
    p3 = copy(pos_vec)
    p1[pair_dir] = 3*p1[pair_dir] - 2
    p2[pair_dir] = 3*p2[pair_dir] - 1
    p3[pair_dir] = 3*p3[pair_dir]

    return [(pos[1] - 1, _linear_ind_simple_lattice(Tuple(p1), dims_ll)), 
            (pos[1] - 1, _linear_ind_simple_lattice(Tuple(p2), dims_ll)),
            (pos[1] - 1, _linear_ind_simple_lattice(Tuple(p3), dims_ll))]
end


function child_nodes(net::TrenaryNetwork{L}, pos::Tuple{Int, Int}) where {L<:SimpleLattice{1}}
    check_valid_position(net, pos)
    pos[1] == 0 && (return nothing)

    return [(pos[1] - 1, 3*pos[2] - 2), (pos[1] - 1, 3*pos[2] - 1), (pos[1] - 1, 3*pos[2])]
end

function index_of_child(net::TrenaryNetwork, pos_child::Tuple{Int,Int})
    pair_dir = mod(pos_child[1], dimensionality(net)) + 1
    pos_vec = _coordinate_simple_lattice(pos_child[2], dimensions(net, pos_child[1]))
    return mod1(pos_vec[pair_dir], 3)
end


index_of_child(::TrenaryNetwork{L}, pos_child::Tuple{Int,Int}) where{L<:SimpleLattice{1}} = mod1(pos_child[2], 3)


function adjacency_matrix(net::TrenaryNetwork, l::Int)
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

#=
function adjacency_matrix(net::TrenaryNetwork{L}, l::Int64) where{L<:SimpleLattice{1}}
    l == number_of_layers(net) && return nothing
	n_this = number_of_tensors(net, l)
	n_next = number_of_tensors(net, l+1)
	pos_this = collect(1:n_this)
	I = repeat(collect(1:n_next), 2)
	J = vcat(pos_this[1:2:end], pos_this[2:2:end])
	return sparse(I,J,repeat([1], n_this), n_next, n_this)
end
=#