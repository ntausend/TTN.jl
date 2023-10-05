
# Needs to have the same backend as the abstract latice
abstract type AbstractNetwork{L<:AbstractLattice} end

# generic network type, needs the connectivity matrices for connecting layers
# only allows for connectivity between adjacent layers...
struct GenericNetwork{L<:AbstractLattice} <: AbstractNetwork{L}
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

# calculates the dimensionality of a layer 
# this is not working...
#=
function dimensionality_reduced(net::AbstractNetwork, l::Int)
    dim_red = filter(isone, size(lattice(net, l)))
    D = isempty(dim_red) ? dimensionality(net) : dimensionality(net) - sum(dim_red)
	return D
end
=#

# sitindices, only relevant for the ITensors case
siteinds(net::AbstractNetwork) = siteinds(physical_lattice(net))

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


# length of the network == total number of sites included in the network
#Base.length(net::AbstractNetwork) = sum(map(jj -> length(lattice(net,jj)), 0:number_of_layers(net)))

"""
	internal_index_of_legs(net::AbstractNetwork, pos::Tuple{Int, Int})

Returns a list of indices associated to the legs of a tensor located at `pos`

## Arguments:

---

- net::AbstractNetwork
- pos::Tuple{Int,Int}: Position of Tensor in the Network

## Returns:

---

For every leg of the tensor it returns a unique index. The indices are increasing
inside one layer (the physical legs are numbered 1, 2, 3, ...., N_phys) such that
the indices of the child legs are adjacent but the prarent index is shifted.

## Example:

---

```
	net = BinaryChainNetwork(2) # creates a network with two virtual layers
	TTNKit.internal_index_of_legs(net, (1,1)) # returns [1,2,5]
	TTNkit.internal_index_of_legs(net, (1,2)) # returns [3,4,6]
	TTNKit.internal_index_of_legs(net, (2,1)) # returns [5,6,7]
```

"""
function internal_index_of_legs(net::AbstractNetwork, pos::Tuple{Int,Int})
	number_of_childs_prev_layers = mapreduce(+,1:pos[1]-1, init = 0) do (ll)
			mapreduce(+, 1:number_of_tensors(net, ll), init = 0) do (pp)
				return number_of_child_nodes(net, (ll,pp))
			end
		end
	n_shift_1 = number_of_childs_prev_layers
	n_shift_2 = length(lattice(net, pos[1]-1)) + number_of_childs_prev_layers
	idx_parent = pos[2] + n_shift_2
	return vcat([jj + n_shift_1 for (_,jj) in child_nodes(net,pos)], idx_parent)
end


physical_coordinates(net::AbstractNetwork) = coordinates(physical_lattice(net))

spacetype( ::Type{<:AbstractNetwork{L}}) where{L} = spacetype(L)
sectortype(::Type{<:AbstractNetwork{L}}) where{L} = sectortype(L)
spacetype(net::AbstractNetwork)  = spacetype(typeof(net))
sectortype(net::AbstractNetwork) = sectortype(typeof(net))

# checking if position is valid
function check_valid_position(net::AbstractNetwork, pos::Tuple{Int, Int})
	l, p = pos
	if !(0 ≤ l ≤ number_of_layers(net)) || !(0 < p ≤ number_of_tensors(net, l))
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
	check_valid_position(net, pos2)
	
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

"""
	connecting_path(net::AbstractNetwork, pos1::Tuple{Int, Int}, pos2::Tuple{Int, Int})

Returns the path through the network connecting `pos1` and `pos2`, excluding `pos1`.
If `pos1 == pos2`, the result is empty. I.e. if the network is a standart Binary tree
with 
```
	       (3,1)
	(2,1)        (2,2)
(1,1), (1,2), (1,3), (1,4) 
```

the path connecting node `(1,1)` with node `(1,2)` would be:
```
(2,1) -> (1,2)
```
while `(1,1)` and `(1,3)` would result in 
```
(2,1) -> (3,1) -> (2,2) -> (1,3)
```
Instead, `(1,3)` to `(1,1)` would result in
```
(2,2) -> (3,1) -> (2,1) -> (1,1)
```

Inputs:
- net: Network describing the architecture.
- pos1: Initial node to start the path, will be **excluded** in the final result
- pos2: Target node terminating the path, will be **included** in the final result

"""
function connecting_path(net::AbstractNetwork, pos1::Tuple{Int, Int}, pos2::Tuple{Int, Int})
	if(pos1[1] < pos2[1])
		path = _connecting_path(net, pos1, pos2)
		return path[2:end]
	else
		path = _connecting_path(net, pos2, pos1)
		return reverse(path)[2:end]
	end	
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
	return GenericNetwork{typeof(lat_vec[1])}(adjmats, lat_vec)
end

struct NodeIterator
	net::AbstractNetwork
end

Base.length(nit::NodeIterator) = mapreduce(l -> length(l), +, init = 0, lattices(nit.net)[2:end])

function Base.iterate(::NodeIterator)
	pos = (1, 1)
	return (pos, pos)
end

function Base.iterate(nit::NodeIterator, state)
	#state == n_layers && return nothing
	state[1] == number_of_layers(nit.net) && return nothing

	if state[2] == number_of_tensors(nit.net, state[1])
		pos = (state[1] + 1, 1)
	else
		pos = (state[1], state[2] + 1)
	end
	return (pos, pos)
end

function Base.iterate(itr::Iterators.Reverse{<:NodeIterator})
	pos = (number_of_layers(itr.itr.net), 1)
	return (pos, pos)
end

function Base.iterate(itr::Iterators.Reverse{<:NodeIterator}, state)
	state == (1,1) && return nothing	
	
	if(state[2] == 1)
		n_l = state[1] - 1
		n_t = number_of_tensors(itr.itr.net, n_l)
		pos = (n_l, n_t)
	else
		pos = (state[1], state[2] - 1)
	end

	return (pos, pos)
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, net::TTNKit.AbstractNetwork)
    g = create_group(parent, name)
    for (num_lat,lat) in enumerate(net.lattices)
        name_lat = "lattice_"*string(num_lat)
        write(g, name_lat, lat)
    end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{TTNKit.AbstractNetwork})
    g = open_group(parent, name)

    lattices = map(keys(g)) do name_lattice
        read(g, name_lattice, TTNKit.AbstractLattice)
    end
    
    return TTNKit.BinaryNetwork{typeof(lattices[1])}(lattices)
end
