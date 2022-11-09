
#abstract type AbstractNetwork{D, S<:IndexSpace, I<:Sector} end
abstract type AbstractNetwork{L<:AbstractLattice} end

# generic network type, needs the connectivity matrices for connecting layers
# only allows for connectivity between adjacent layers...
#struct Network{D, S<:IndexSpace, I<:Sector} <: AbstractNetwork{D,S,I}
struct Network{L<:AbstractLattice} <: AbstractNetwork{L}
	# adjacency matrix per layer, first is physical connectivity 
	# to first TTN Layer
    connections::Vector{SparseMatrixCSC{Int, Int}}
	#lattices per layer, first is physical layer
	#lattices::Vector{AbstractLattice{D,S,I}}
	lattices::Vector{L}
end

# dimensionality of network
dimensionality(::Type{<:AbstractNetwork{L}}) where L  = dimensionality(L)
dimensionality(net::AbstractNetwork) = dimensionality(typeof(net))

# returning the lattices of the network, 1 is physical, and 2:n_layers are the virtuals
lattices(net::AbstractNetwork) = net.lattices
# returning the l-th layer. l is here defined with respect to the first **virtual** layer
lattice(net::AbstractNetwork, l::Int) = lattices(net)[l + 1]

# returns the physical lattice
physical_lattice(net::AbstractNetwork) = lattice(net, 0)

# returning the number of **virtual** layers
number_of_layers(net::AbstractNetwork) = length(lattices(net)) - 1

# returning the number of nodes living on the `l` virtual layer
number_of_tensors(net::AbstractNetwork, l::Int) = number_of_sites(lattice(net,l))
# returning the total number of nodes in all virtual layers
number_of_tensors(net::AbstractNetwork) = sum(map(l -> number_of_tensors(net, l), 1:number_of_layers(net)))

# adjacency matrix associated with connecting the `l` and `l+1` virtual layer. l = 0 should
# define the connectivitiy of the physical layer with the first virtual layer
adjacency_matrix(net::AbstractNetwork, l::Int) = net.connections[l + 1]

# returns the `p[2]` node in the `p[1]` layer
node(net::AbstractNetwork, p::Tuple{Int,Int}) = node(lattice(net,p[1]), p[2])

# factory function for creating hilbertspaces out of the node types on the different layers, may be removed inthe future
hilbertspace(net::AbstractNetwork, p::Tuple{Int,Int}, sectors) = hilbertspace(node(lattice(net,p[1]),p[2]), sectors)

# dimensions of the `l`-th layer
dimensions(net::AbstractNetwork, l::Int) = size(lattice(net, l))

# number of physical sites
number_of_sites(net::AbstractNetwork) = number_of_sites(physical_lattice(net))

physical_coordinates(net::AbstractNetwork) = coordinates(physical_lattice(net))

TensorKit.spacetype( ::Type{<:AbstractNetwork{L}}) where{L} = spacetype(L)
TensorKit.sectortype(::Type{<:AbstractNetwork{L}}) where{L} = sectortype(L)
TensorKit.spacetype(net::AbstractNetwork)  = spacetype(typeof(net))
TensorKit.sectortype(net::AbstractNetwork) = sectortype(typeof(net))

# checking if position is valid
function check_valid_position(net::AbstractNetwork, pos::Tuple{Int, Int})
	l, p = pos
	if !(0 ≤ l ≤ number_of_layers(net)) && (0 < p ≤ number_of_tensors(net, l))
		throw(BoundsError(net,pos))
	end
end

"""
	index_of_child(net::AbstractNetwork, pos_child::Tuple{Int,Int})

Returns the index of a child in the child list of its parent.
Needed for uniquly identifying the leg associated to this child
in the parent tensor.

"""
function index_of_child(net::AbstractNetwork, pos_child::Tuple{Int, Int})
	pos_parent = parent_node(net,pos_child)
	idx_child = findfirst(x -> all(x .== pos_child), child_nodes(net, pos_parent))
	return idx_child
end



"""
	parent_node(net::AbstractNetwork, pos::Tuple{Int, Int})

Returns the (unique) parent node of the given position tuple. 
If `pos` is the top node, `nothing` is returned.

## Inputs:
- `net`: Network to find the parent node
- `pos`: Tuple consisting of the childs layer and interlayer position
"""
function parent_node(net::AbstractNetwork, pos::Tuple{Int, Int})
	check_valid_position(net, pos)
	
	pos[1] == number_of_layers(net) && (return nothing)
		
	adjMat = adjacency_matrix(net, pos[1])
	idx_parent = findall(!iszero, adjMat[:,pos[2]])
	length(idx_parent) > 1 && error("Number of parents not exactly 1. Illdefined Tree Tensor Network")
	return (pos[1] + 1, idx_parent[1])
end


"""
	child_nodes(net::AbstractNetwork, pos::Tuple{Int, Int})

Returns an array representing all childs. In case of node beeing in the lowest
layer, it returns a list of the form (0, p) representing the physical site.

Inputs: See `parent_node(net::AbstractNetwork, pos::Tuple{Int, Int})`
"""
function child_nodes(net::AbstractNetwork, pos::Tuple{Int, Int})
	check_valid_position(net, pos)
	pos[1] == 0 && (return nothing)

	adjMat = adjacency_matrix(net, pos[1]-1)
	inds = findall(!iszero, adjMat[pos[2],:])
	return [(pos[1] - 1, idx) for idx in inds]
end

# returns the number of child nodes
function number_of_child_nodes(net::AbstractNetwork, pos::Tuple{Int, Int})
	return length(child_nodes(net, pos))
end

"""
	split_index(net::AbstractNetwork, pos::Tuple{Int,Int}, idx::Int)

This function returns ((idx),(I\\idx)) with I being the complete index
set of a given tensor at `pos`. This function is needed to perform perumations
such that idx is singled out in the domain/codomain and all other indices are
in the other space.
"""
function split_index(net, pos, idx)
	allinds  = 1:(1+number_of_child_nodes(net, pos))
	res_inds = Tuple(deleteat!(collect(allinds), idx))
	return Tuple(idx), res_inds
end

# function returning all nodes from `pos1` to `pos2` by assuming
# that the layer of `pos1` is lower or equal than `pos2`
function _connecting_path(net::AbstractNetwork, pos1::Tuple{Int, Int}, pos2::Tuple{Int, Int})
	check_valid_position(net, pos1)
	check_valid_position(net, pos1)
	
	# first assume l1 ≤ l2, p1 < p2 for simplicity
	# later to the general thing
	# fast excit in case both are equal
	all(pos1 .== pos2) && (return Vector{Tuple{Int64, Int64}}())

	pos_cur = pos1 
	# first bring l1 and l2 to the same layer
	path_delta = Vector{Tuple{Int64, Int64}}(undef, pos2[1] - pos1[1])
	for jj in 1:(pos2[1]-pos1[1])
		pos_cur = parent_node(net, pos_cur)
		path_delta[jj] = pos_cur
	end
	# if pos_cur == pos2, we can exit
	pos_cur == pos2 && return vcat(pos1, path_delta)
	posl = pos_cur
	

	# now seach for the same parent 
	pathl = Vector{Tuple{Int64, Int64}}()
	pathr = Vector{Tuple{Int64, Int64}}()
	posr = pos2
	while true
		posl = parent_node(net, posl)
		posr = parent_node(net, posr)
		#TODO -> smarter?
		if(posl == posr)
			push!(pathr, posr)
			break
		end
		!isnothing(posl) &&	push!(pathl, posl)
		push!(pathr, posr)
	end
	path = vcat(pos1, path_delta, pathl, reverse(pathr), pos2)
	return path
end

function connecting_path(net::AbstractNetwork, pos1::Tuple{Int, Int}, pos2::Tuple{Int, Int})
	if(pos1[1] < pos2[1])
		path = _connecting_path(net, pos1, pos2)
		return path[2:end]
	else
		path = _connecting_path(net, pos2, pos1)
		return reverse(path)[2:end]
	end	
end


function Base.iterate(::AbstractNetwork)
	pos = (1, 1)
	return (pos, pos)
end

function Base.iterate(net::AbstractNetwork, state)
	#state == n_layers && return nothing
	state[1] == number_of_layers(net) && return nothing

	if state[2] == number_of_tensors(net, state[1])
		pos = (state[1] + 1, 1)
	else
		pos = (state[1], state[2] + 1)
	end
	return (pos, pos)
end

function Base.iterate(itr::Iterators.Reverse{<:AbstractNetwork})
	pos = (number_of_layers(itr.itr), 1)
	return (pos, pos)
end

function Base.iterate(itr::Iterators.Reverse{<:AbstractNetwork}, state)
	state == (1,1) && return nothing	
	
	if(state[2] == 1)
		n_l = state[1] - 1
		n_t = number_of_tensors(itr.itr, n_l)
		pos = (n_l, n_t)
	else
		pos = (state[1], state[2] - 1)
	end

	return (pos, pos)
end

eachlayer(net::AbstractNetwork) = 1:number_of_layers(net)
eachindex(net::AbstractNetwork,l::Int) = eachindex(lattice(net,l))




"""
	NetworkBinaryOneDim(n::Int, bonddims::Vector{Int}, local_dim::Int)

Defines a one dimensional binary tree tensor network structure where every
odd 2j-1 and even site 2j for j≥1 are connected by the next layer tensor.

- `n`: Defines the number of layers, the lowest layer then has ``2^{n-1}`` tensors.
- `bonddims`: Defines the maximal bond dimension for connecting ajdacent layers. `bonddims[1]` then defines the connectivity between the lowest and the next layer, etc.
"""
function CreateBinaryChainNetwork(n_layers::Int, local_dim::Int)
	#bnddm = correct_bonddims(bonddims, n)
	
	#@assert length(bonddims) == n_layers-1
	adjmats = Vector{SparseMatrixCSC{Int64, Int64}}(undef, n_layers)

	lat_vec = Vector{SimpleLattice{1}}(undef, n_layers+1)
	#bnddim_comp = vcat(local_dim, bonddims)

	lat_vec[1] = Chain(2^(n_layers), TrivialNode; local_dim = local_dim)
	vnd_type = nodetype(lat_vec[1])

	for jj in n_layers:-1:1
		n_this = 2^(jj)
		n_next = 2^(jj-1)

		adjMat = spzeros(n_next, n_this)
		for ll in 1:2:n_this
			idx_next = div(ll+1,2)
			adjMat[idx_next, ll]   = 1#bnddim_layer
			adjMat[idx_next, ll+1] = 1#bnddim_layer
		end
		adjmats[n_layers - jj + 1] = adjMat
		
		if(jj<n_layers)
			lat_vec[n_layers - jj + 1] = Chain(n_this, vnd_type)
		end
	end
	lat_vec[end] = Chain(1,vnd_type)
	
	#return Network{1, spacetype(lat_vec[1]), sectortype(lat_vec[1])}(adjmats, lat_vec)
	return Network{typeof(lat_vec[1])}(adjmats, lat_vec)
end
