struct BinaryNetwork{D} <: AbstractNetwork{D}
    tensors_per_layer::Vector{Int}
    bonddims::Vector{Int}
    lattice::BinaryLattice{D}
    layer_dimensions::Vector{NTuple{D,Int}}
end

function BinaryNetwork(dimensions::NTuple{D,Int}, bonddims::Vector{Int}, localDim::Int; field = ComplexSpace) where{D}
    lat = BinaryLattice(dimensions, localDim; field = field)
    n_layer = sum(Int64.(log2.(dimensions)))
    # we need a strategy for extending the bonddims to the correct length...
    @assert n_layer == length(bonddims) + 1

    tensors_per_layer = [2^(n_layer - jj) for jj in 1:n_layer]

    layer_dimensions = Vector{NTuple{D,Int}}(undef, n_layer)
    tp_c = vcat(size(lat)...)

    for jj in 1:n_layer
        pair_dir = mod(jj, D) + 1
        red = zeros(Int64, D)
        red[pair_dir] = 2
        tp_c = tp_c .- red
        tp_c[tmp .== 0] .= 1
        layer_dimensions[jj] = Tuple(tp_c)
    end

    return BinaryNetwork(tensors_per_layer, bonddims, lat, layer_dimensions)
end

# number of child nodes of position. For binary networks this is a constant of 2
n_childNodes(::BinaryNetwork, ::Tuple{Int,Int}) = 2

const BinaryChainNetwork = BinaryNetwork{1}
const BinaryRectangularNetwork = BinaryNetwork{2}

function BinaryChainNetwork(number_of_layers::Int, bonddims::Vector{Int}, local_dim::Int; kwargs...)
    @assert number_of_layers == length(bonddims) + 1
    tensors_per_layer = [2^(number_of_layers - jj) for jj in 1:number_of_layers]
    lat = BinaryChain(2^number_of_layers, local_dim; kwargs...)

    return BinaryChainNetwork(tensors_per_layer, bonddims, lat, Tuple.(tensors_per_layer)) 
end

function BinaryRectangularNetwork(number_of_layers::Int, bonddims::Vector{Int}, local_dim::Int; kwargs...)
    @assert number_of_layers == length(bonddims) + 1
    tensors_per_layer = [2^(number_of_layers - jj) for jj in 1:number_of_layers]
   
    # number of layers is always representitive as n_l = 2*n + r
    # the number of sites in x direction is 2^((2n + r + 1) / 2) since
    # we start with a pairing in x direction. In case r == 1 this gives
    # an extra doubling in x. On the other hand, the number of sites in y direction
    # are given by 2^n. If r == 0 the both numbers conincidend and we have a square lattice.

    n_x = 2^(div(number_of_layers + 1, 2))
    n_y = 2^(div(number_of_layers, 2))
    lat = BinaryRectangle(n_x, n_y, local_dim; kwargs...)

    layer_dimensions = Vector{Tuple{Int, Int}}(undef, number_of_layers)
    layer_dimensions[end] = (1,1)
    tp_c = vcat(size(lat)...)
    for jj in 1:number_of_layers-1
        red = iseven(jj) ? (0,2) : (2,0)
        tp_c = tp_c .- red
        tp_c[tp_c .== 0] .= 1
        layer_dimensions[jj] = Tuple(tp_c)
    end
    return BinaryRectangularNetwork(tensors_per_layer, bonddims, lat, layer_dimensions)
end

# exact formular for this kind of network
n_tensors(net::BinaryNetwork) = 2^(n_layers(net)) - 1

# dimensions of the lattice on the n-th layer
dimensions(net::BinaryNetwork, l::Int64) = net.layer_dimensions[l]


function parentNode(net::BinaryNetwork, pos::Tuple{Int, Int})
    @assert check_valid_pos(net, pos)
    pos[1] == n_layers(net) && (return nothing)
    # check if paring is along x or y direction in the next step
    # even layers are paired along the x direction, odd layers along the y direction
    pair_dir = mod(pos[1], dimensionality(net)) + 1
    # unroll the linear index
    pos_vec = vcat(_to_coordinate(pos[2], dimensions(net, pos[1]))...)

    # the coordinate of the pairing dimension is given by either
    #           p_j = 2n - 1 or 2n
    # both coordinates are maped to the parent coordinate p̃ = n
    # which are then converted back to the linear index of the next layer.
    pos_vec[pair_dir] = div(pos_vec[pair_dir] + 1, 2)

    return (pos[1] + 1, _to_linear_ind(Tuple(pos_vec), dimensions(net, pos[1] + 1)))
end

function parentNode(net::BinaryChainNetwork, pos::Tuple{Int, Int})
    @assert check_valid_pos(net, pos)
    pos[1] == n_layers(net) && (return nothing)

    return(pos[1] + 1, div(pos[2] + 1,2))
end


function childNodes(net::BinaryNetwork, pos::Tuple{Int, Int})
    @assert check_valid_pos(net, pos)
    
    # do the revert operation as for the parent nodes
    # pairing of this layer, given by the pairing direction of
    # the previous layer
    pair_dir = mod(pos[1] - 1, dimensionality(net)) + 1
    
    # getting the coordinates inside the current layer
    pos_vec = vcat(_to_coordinate(pos[2], dimensions(net, pos[1]))...)

    # getting the dimensions of the lower layer.
    dims_ll = pos[1] == 1 ? size(lattice(net)) : dimensions(net, pos[1] - 1)

    p1 = copy(pos_vec)
    p2 = copy(pos_vec)
    p1[pair_dir] = 2*p1[pair_dir] - 1 
    p2[pair_dir] = 2*p2[pair_dir]

    return [(pos[1] - 1, _to_linear_ind(Tuple(p1), dims_ll)), (pos[1] - 1, _to_linear_ind(Tuple(p2), dims_ll))]
end

function childNodes(net::BinaryChainNetwork, pos::Tuple{Int, Int})
    @assert check_valid_pos(net, pos)
    return [(pos[1] - 1, 2*pos[2] - 1), (pos[1] - 1, 2*pos[2])]
end

function index_of_child(net::BinaryNetwork, pos_child::Tuple{Int,Int})
    pair_dir = mod(pos_child[1], dimensionality(net)) + 1
    pos_vec = _to_coordinate(pos_child[2], dimensions(net, pos_child[1]))
    return mod1(pos_vec[pair_dir], 2)
end

index_of_child(::BinaryChainNetwork, pos_child::Tuple{Int,Int}) = mod1(pos_child[2], 2)


function adjacencyMatrix(net::BinaryNetwork, l::Int)
    l == n_layers(net) && return nothing
	n_this = n_tensors(net, l)
	n_next = n_tensors(net, l+1)
    I  = zeros(Int64, n_this)
    J  = collect(1:n_this)

    for jj in 1:n_this
        parent_idx = parentNode(net, (l,jj))
        I[jj] = parent_idx[2]
    end
    return sparse(I,J, repeat([1], n_this), n_next, n_this)
end

function adjacencyMatrix(net::BinaryChainNetwork, l::Int64)
    l == n_layers(net) && return nothing
	n_this = n_tensors(net, l)
	n_next = n_tensors(net, l+1)
	pos_this = collect(1:n_this)
	I = repeat(collect(1:n_next), 2)
	J = vcat(pos_this[1:2:end], pos_this[2:2:end])
	return sparse(I,J,repeat([1], n_this), n_next, n_this)
end