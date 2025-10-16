using TTN

import TTN: SimpleLattice, AbstractNetwork, AbstractNode, Index, ITensorNode, TrivialNode, nodetype, check_valid_position, _coordinate_simple_lattice, _linear_ind_simple_lattice

function _dims_to_n_layer_six(dims::NTuple{D, Int}) where D
    return 4
end

struct SixBySixNetwork{L<:SimpleLattice} <: AbstractNetwork{L}
    lattices::Vector{L}
end

function SixBySixNetwork(dimensions::NTuple{D,Int}, nd::Type{<:AbstractNode}; kwargs...) where{D}
    n_layer = _dims_to_n_layer_six(dimensions)
    lat_vec = Vector{SimpleLattice{D}}(undef, n_layer + 1)

    dimensionsc = vcat(dimensions...)
    # first dimension must be largest, second second largest etc..
    # this is required due to our pairing
    if !(sort(dimensionsc) == reverse(dimensionsc))
        msg = "Only Lattices with first dimension being largest, second being second largest etc are suppported for SixBySixNetworks. Dimensions: $dimensions"   
        throw(NotSupportedException(msg))
    end
    lat_vec[1] = SimpleLattice(dimensions, nd; kwargs...)

    vnd_type = nodetype(lat_vec[1])
    for jj in 2:n_layer+1
        # this is not working...
        D_actual = D #- sum(dimensionsc[2:end][dimensionsc[2:end] .== 1])
        pair_dir  = mod1(jj-1, D_actual)
        dimensionsc[pair_dir] = jj <= 3 ? div(dimensionsc[pair_dir],3) : div(dimensionsc[pair_dir],2)
        
        #dimensionsc[dimensionsc.==0] .= 1
        lat = SimpleLattice(Tuple(dimensionsc), vnd_type)
        lat_vec[jj] = lat
        
        # pairing direction of the next layer
    end
    
    #return SixBySixNetwork{D, spacetype(vnd_type), sectortype(vnd_type)}(lat_vec)
    return SixBySixNetwork{typeof(lat_vec[1])}(lat_vec)
end
SixBySixNetwork(dimensions::NTuple; kwargs...) = SixBySixNetwork(dimensions, TrivialNode; kwargs...)

# creation from indices, may be fused with above function?
function SixBySixNetwork(dims::NTuple{D, Int}, indices::Vector{<:Index}) where{D}
    @assert prod(dims) == length(indices)
    n_layer = _dims_to_n_layer_six(dims)
    lat_vec = Vector{SimpleLattice{D}}(undef, n_layer + 1)

    dimensionsc = vcat(dims...)
    # first dimension must be largest, second second largest etc..
    # this is required due to our pairing
    if !(sort(dimensionsc) == reverse(dimensionsc))
        msg = "Only Lattices with first dimension being largest, second being second largest etc are suppported for SixBySixNetworks. Dimensions: $dimensions"   
        throw(NotSupportedException(msg))
    end
    lat_vec[1] = SimpleLattice(dims, indices)
    vnd_type = nodetype(lat_vec[1])

    for jj in 2:n_layer+1
        pair_dir  = mod1(jj-1, D)
        if dimensionsc[pair_dir] == 1
          pair_dir = mod(pair_dir+1, 2) 
        end
        dimensionsc[pair_dir] = jj <= 3 ? div(dimensionsc[pair_dir],3) : div(dimensionsc[pair_dir],2)
        dimensionsc[dimensionsc.==0] .= 1

        lat = SimpleLattice(Tuple(dimensionsc), vnd_type)
        lat_vec[jj] = lat
        # pairing direction of the next layer
    end
    
    #return SixBySixNetwork{D, spacetype(vnd_type), sectortype(vnd_type)}(lat_vec)
    return SixBySixNetwork{typeof(lat_vec[1])}(lat_vec)
end
# creation from ITensorNode with type specifier
function SixBySixNetwork(dims::NTuple{D, Int}, nd::Type{<:ITensorNode}, type::AbstractString; kwargs...) where{D}
    indices = siteinds(type, prod(dims); kwargs...)
    return SixBySixNetwork(dims, indices)
end
SixBySixNetwork(dims, type; kwargs...) = SixBySixNetwork(dims, ITensorNode, type; kwargs...)


# number of child nodes of position. For six networks this is a constant of 2
function number_of_child_nodes(::SixBySixNetwork, pos::Tuple{Int,Int})
  first(pos) <=2 && return 3
  return 2
end
# exact formular for this kind of network
number_of_tensors(net::SixBySixNetwork) = 19 # 3 * 2^(number_of_layers(net)-2) - 1


#const SixBySixChainNetwork{S<:IndexSpace, I<:Sector} = SixBySixNetwork{1, S, I}
const SixBySixChainNetwork{L<:SimpleLattice{1}} = SixBySixNetwork{L}

function parent_node(net::SixBySixNetwork, pos::Tuple{Int, Int})
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
    if pos[1] < 2
      pos_vec[pair_dir] = div(pos_vec[pair_dir] + 2, 3)
    else
      pos_vec[pair_dir] = div(pos_vec[pair_dir] + 1, 2)
    end

    return (pos[1] + 1, _linear_ind_simple_lattice(Tuple(pos_vec), dimensions(net, pos[1] + 1)))
end

# function parent_node(net::SixBySixNetwork{L}, pos::Tuple{Int, Int}) where {L<:SimpleLattice{1}}
#     check_valid_position(net, pos)
#     pos[1] == number_of_layers(net) && (return nothing)
#
#     return(pos[1] + 1, div(pos[2] + 1,2))
# end


function child_nodes(net::SixBySixNetwork, pos::Tuple{Int, Int})
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

    if pos[1] < 3
      p1 = copy(pos_vec)
      p2 = copy(pos_vec)
      p3 = copy(pos_vec)
      p1[pair_dir] = 3*p1[pair_dir] - 2 
      p2[pair_dir] = 3*p2[pair_dir] - 1 
      p3[pair_dir] = 3*p3[pair_dir]

      return [(pos[1] - 1, _linear_ind_simple_lattice(Tuple(p1), dims_ll)), 
              (pos[1] - 1, _linear_ind_simple_lattice(Tuple(p2), dims_ll)),
              (pos[1] - 1, _linear_ind_simple_lattice(Tuple(p3), dims_ll))]
    else
      p1 = copy(pos_vec)
      p2 = copy(pos_vec)
      p1[pair_dir] = 2*p1[pair_dir] - 1 
      p2[pair_dir] = 2*p2[pair_dir]

      return [(pos[1] - 1, _linear_ind_simple_lattice(Tuple(p1), dims_ll)), 
              (pos[1] - 1, _linear_ind_simple_lattice(Tuple(p2), dims_ll))]
    end
end

# function child_nodes(net::SixBySixNetwork{L}, pos::Tuple{Int, Int}) where {L<:SimpleLattice{1}}
#     check_valid_position(net, pos)
#     pos[1] == 0 && (return nothing)
#
#     return [(pos[1] - 1, 2*pos[2] - 1), (pos[1] - 1, 2*pos[2])]
# end

function index_of_child(net::SixBySixNetwork, pos_child::Tuple{Int,Int})
    pair_dir = mod(pos_child[1], dimensionality(net)) + 1
    pos_vec = _coordinate_simple_lattice(pos_child[2], dimensions(net, pos_child[1]))
    pos_child[1] < 2 && return mod1(pos_vec[pair_dir], 3)
    return mod1(pos_vec[pair_dir], 2)
end

# index_of_child(::SixBySixNetwork{L}, pos_child::Tuple{Int,Int}) where{L<:SimpleLattice{1}} = mod1(pos_child[2], 2)


function adjacency_matrix(net::SixBySixNetwork, l::Int)
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

function adjacency_matrix(net::SixBySixNetwork{L}, l::Int64) where{L<:SimpleLattice{1}}
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
function internal_index_of_legs(net::SixBySixChainNetwork, pos::Tuple{Int,Int})
    n_layers = number_of_layers(net)
    number_of_childs_prev_layers = 2^(n_layers - pos[1] + 1)
    n_shift_1 = 2^(pos[2] - 1) + number_of_childs_prev_layers 
    n_shift_2 = 2^(n_layers - pos[2] + 2)
	return [1 + n_shift_1, 2 + n_shift_1, n_shift_2 + pos[2]]
end
=#
