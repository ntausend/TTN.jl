struct BinaryNetwork{D, S<:IndexSpace, I<:Sector} <: AbstractNetwork{D, S, I}
    lattices::Vector{BinaryLattice{D, S, I}}
end

function BinaryNetwork(dimensions::NTuple{D,Int}, nd::Type{<:AbstractNode}; kwargs...) where{D}
    n_layer = _dims_to_n_layer(dimensions)
    lat_vec = Vector{BinaryLattice{D}}(undef, n_layer + 1)

    dimensionsc = vcat(dimensions...)
    lat_vec[1] = BinaryLattice(dimensions, nd; kwargs...)

    vnd_type = VirtualNode(lat_vec[1])

    for jj in 2:n_layer+1
        pair_dir  = mod1(jj-1, D)
        dimensionsc[pair_dir] = div(dimensionsc[pair_dir],2)
        dimensionsc[dimensionsc.==0] .= 1

        lat = BinaryLattice(Tuple(dimensionsc), vnd_type)
        lat_vec[jj] = lat
        # pairing direction of the next layer
    end
    
    return BinaryNetwork{D, space(vnd_type), sectortype(vnd_type)}(lat_vec)
end
function BinaryNetwork(dimensions::NTuple; kwargs...) 
    return BinaryNetwork(dimensions, TrivialNode; kwargs...)
end


# number of child nodes of position. For binary networks this is a constant of 2
n_childNodes(::BinaryNetwork, ::Tuple{Int,Int}) = 2
# exact formular for this kind of network
n_tensors(net::BinaryNetwork) = 2^(n_layers(net)) - 1


const BinaryChainNetwork{S<:IndexSpace, I<:Sector} = BinaryNetwork{1, S, I}
function BinaryChainNetwork(number_of_layers::Int, nd::Type{<:AbstractNode}; kwargs...)
    tensors_per_layer = [2^(number_of_layers - jj) for jj in 0:number_of_layers]
    phys_lat = BinaryChain(tensors_per_layer[1], nd; kwargs...)
    nvd_type = VirtualNode(phys_lat)
    lat_vec = map(tensors_per_layer) do (nn)
        BinaryChain(nn, nvd_type)
    end
    lat_vec[1] = phys_lat
    return BinaryChainNetwork{space(nvd_type), sectortype(nvd_type)}(lat_vec)
end
BinaryChainNetwork(number_of_layers::Int; kwargs...) = BinaryChainNetwork(number_of_layers, TrivialNode; kwargs...)


function BinaryRectangularNetwork(number_of_layers::Int, nd::Type{<:AbstractNode}; kwargs...)
    
    # calculate the physical dimensions and use the fall back function
    # number of layers is always representitive as n_l = 2*n + r
    # the number of sites in x direction is 2^((2n + r + 1) / 2) since
    # we start with a pairing in x direction. In case r == 1 this gives
    # an extra doubling in x. On the other hand, the number of sites in y direction
    # are given by 2^n. If r == 0 the both numbers conincidend and we have a square lattice.

    n_x = 2^(div(number_of_layers + 1, 2))
    n_y = 2^(div(number_of_layers, 2))
    return BinaryNetwork((n_x, n_y), nd; kwargs...)
end
BinaryRectangularNetwork(number_of_layers::Int; kwargs...) = BinaryRectangularNetwork(number_of_layers, TrivialNode; kwargs...)


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
    
    pos[1] == 0 && (return nothing)

    # do the revert operation as for the parent nodes
    # pairing of this layer, given by the pairing direction of
    # the previous layer
    pair_dir = mod(pos[1] - 1, dimensionality(net)) + 1
    
    # getting the coordinates inside the current layer
    pos_vec = vcat(_to_coordinate(pos[2], dimensions(net, pos[1]))...)

    # getting the dimensions of the lower layer.
    #dims_ll = pos[1] == 1 ? size(lattice(net)) : dimensions(net, pos[1] - 1)
    dims_ll = dimensions(net, pos[1] - 1)

    p1 = copy(pos_vec)
    p2 = copy(pos_vec)
    p1[pair_dir] = 2*p1[pair_dir] - 1 
    p2[pair_dir] = 2*p2[pair_dir]

    return [(pos[1] - 1, _to_linear_ind(Tuple(p1), dims_ll)), (pos[1] - 1, _to_linear_ind(Tuple(p2), dims_ll))]
end

function childNodes(net::BinaryChainNetwork, pos::Tuple{Int, Int})
    @assert check_valid_pos(net, pos)
    pos[1] == 0 && (return nothing)

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


#=
function BinaryNetwork(dimensions::NTuple{D,Int}, maxdim::Int, nodetype::Function; kwargs...) where{D}
    n_layer = _dims_to_n_layer(dimensions)

    # test the sectortype of nodetype
    tmp_nd = nodetype(1,""; kwargs...)
    sec_type = sectortype(tmp_nd)

    if !(sec_type == Trivial)
        error("Symmetrical network needs a state vector or a target QN sector..")
    end

    bonddims_comp = Vector{Int}(undef, n_layer-1)
    bnd_pre = dim(hilbertspace(tmp_nd))
    for ll in 1:n_layer-1
        bonddims_comp[ll] = min(bnd_pre^2, maxdim)
        bnd_pre = bonddims_comp[ll]
    end
    return BinaryNetwork(dimensions, bonddims_comp, nodetype; kwargs...)
end

function BinaryNetwork(dimensions::NTuple{D,Int}, bonddims::Vector{Int}, nodetype::Function; kwargs...) where{D}
    n_layer = _dims_to_n_layer(dimensions)

    # test the sectortype of nodetype
    tmp_nd = nodetype(1,""; kwargs...)
    sec_type = sectortype(tmp_nd)

    if !(sec_type == Trivial)
        error("Symmetrical network needs a state vector or a target QN sector..")
    end
    
    # we need a strategy for extending the bonddims to the correct length...
    @assert n_layer == length(bonddims) + 1
    
    # vector holding the lattices of the different layers
    lat_vec = Vector{BinaryLattice{D}}(undef, n_layer + 1)

    # attach the physical and the trivial dimensions
	bnddim_comp = vcat(dim(hilbertspace(tmp_nd)), bonddims, 1)

    dimensionsc = vcat(dimensions...)

    for jj in 1:n_layer+1
        if jj == 1
            lat = BinaryLattice(Tuple(dimensionsc), nodetype; kwargs...)
        else
            field = get(kwargs, :field, ComplexSpace)
            lat = BinaryLattice(Tuple(dimensionsc), bnddim_comp[jj], field = field)
        end
        lat_vec[jj] = lat
        # pairing direction of the next layer
        pair_dir  = mod1(jj, D)
        dimensionsc[pair_dir] = div(dimensionsc[pair_dir],2)
        dimensionsc[dimensionsc.==0] .= 1
    end
    
    return BinaryNetwork(lat_vec)
end
function BinaryChainNetwork(number_of_layers::Int, bonddims::Vector{Int}, local_dim::Int; kwargs...)
    @assert number_of_layers == length(bonddims) + 1
    tensors_per_layer = [2^(number_of_layers - jj) for jj in 0:number_of_layers]

	bnddim_comp = vcat(local_dim, bonddims, 1)

    lat_vec = map(zip(tensors_per_layer, bnddim_comp)) do (nn, bb)
        BinaryChain(nn, bb; kwargs...)
    end
        
    return BinaryChainNetwork(lat_vec)
end
=#