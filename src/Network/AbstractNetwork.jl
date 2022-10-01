
abstract type AbstractNetwork{D} end

struct Network{D} <: AbstractNetwork{D}
	# tensors per layer
	n_layer::Vector{Int64}
	# bonddimension per layer
	bonddims::Vector{Int64}
	# adjacency matrix per layer
    connections::Vector{SparseMatrixCSC{Int64, Int64}}
	#physical sites -> replace this by a lattice class
	lattice::AbstractLattice{D}
	#lattice::Array{Int64,D}
	#physical dimension
end

dimensionality(net::AbstractNetwork{D}) where D  = D
n_tensors(net::AbstractNetwork, l::Int64) = net.n_layer[l]
n_layers(net::AbstractNetwork) = length(net.n_layer)
n_tensors(net::AbstractNetwork) = sum(map(l -> n_tensors(net, l), 1:n_layers(net)))
adjacencyMatrix(net::AbstractNetwork, l::Int64) = net.connections[l]
bonddim(net::AbstractNetwork, l::Int64) = net.bonddims[l]
lattice(net::AbstractNetwork) = net.lattice
local_dim(net::AbstractNetwork) = local_dim(lattice(net))
adjacencyMatrixPhysicalSites(net::AbstractNetwork) = adjacencyMatrix(lattice(net))

function check_valid_pos(net::AbstractNetwork, pos::Tuple{Int64, Int64})
	l, p = pos
	return (0 < l ≤ n_layers(net)) && (0 < p ≤ n_tensors(net, l))
end

"""
	index_of_child(net::AbstractNetwork, pos_child)

Returns the index of a child in the child list of its parent.
Needed for uniquly identifying the leg associated to this child
in the parent tensor.

"""
function index_of_child(net::AbstractNetwork, pos_child)
	pos_parent = parentNode(net,pos_child)
	idx_child = findfirst(x -> all(x .== pos_child), childNodes(net, pos_parent))
	return idx_child
end


"""
	NetworkBinaryOneDim(n::Int64,bonddims::Vector{Int64})

Defines a one dimensional binary tree tensor network structure where every
odd 2j-1 and even site 2j for j≥1 are connected by the next layer tensor.

- `n`: Defines the number of layers, the lowest layer then has ``2^{n-1}`` tensors.
- `bonddims`: Defines the maximal bond dimension for connecting ajdacent layers. `bonddims[1]` then defines the connectivity between the lowest and the next layer, etc.
"""
function CreateBinaryNetwork(n::Int64,bonddims::Vector{Int64}, local_dim::Int64)
	#bnddm = correct_bonddims(bonddims, n)
	
	@assert length(bonddims) == n-1
	adjmats = Vector{SparseMatrixCSC{Int64, Int64}}(undef, n - 1)
	n_layer = Vector{Int64}(undef, n)
	for jj in n:-1:2
		n_this = 2^(jj-1)
		n_next = 2^(jj-2)

		adjMat = spzeros(n_next, n_this)
		#bnddim_layer = bnddm[n - jj + 1]
		for ll in 1:2:n_this
			idx_next = div(ll+1,2)
			adjMat[idx_next, ll]   = 1#bnddim_layer
			adjMat[idx_next, ll+1] = 1#bnddim_layer
		end
		adjmats[n - jj + 1] = adjMat
		n_layer[n - jj + 1] = n_this
	end
	n_layer[end] = 1

	lattice = CreateBinaryChain(2^n, local_dim)
	#local_inds = collect(1:n_layer[1])
		
	return Network{1}(n_layer, bonddims, adjmats, lattice)
end


"""
	parentNode(net::AbstractNetwork, pos::Tuple{Int64, Int64})

Returns the (unique) parent node of the given position tuple. 
If `pos` is the top node, `nothing` is returned.

## Inputs:
- `net`: Network to find the parent node
- `pos`: Tuple consisting of the childs layer and interlayer position
"""
function parentNode(net::AbstractNetwork, pos::Tuple{Int64, Int64})
	l,p = pos
	
	@assert 0 < l ≤ n_layers(net)
	@assert 0 < p ≤ n_tensors(net, l)

	
	l == n_layers(net) && (return nothing)
		
	adjMat = adjacencyMatrix(net, l)
	idx_parent = findall(!iszero, adjMat[:,p])
	length(idx_parent) > 1 && error("Number of parents not exactly 1. Illdefined Tree Tensor Network")
	return (l + 1, idx_parent[1])
end


"""
	childNodes(net::AbstractNetwork, pos::Tuple{Int64, Int64})

Returns an array representing all childs. In case of node beeing in the lowest
layer, it returns a list of the form (0, p) representing the physical site.

Inputs: See `parent(net::AbstractNetwork, pos::Tuple{Int64, Int64})`
"""
function childNodes(net::AbstractNetwork, pos::Tuple{Int64, Int64})
	l,p = pos
	@assert 0 < l ≤ n_layers(net)
	@assert 0 < p ≤ n_tensors(net, l)

	if (l == 1)
		adjMat = adjacencyMatrixPhysicalSites(net)
	else
		adjMat = adjacencyMatrix(net, l - 1)
	end
	#l == 1 && (return nothing)
	inds = findall(!iszero, adjMat[p,:])
	return [(l - 1, idx) for idx in inds]
end

function n_childNodes(net::AbstractNetwork, pos::Tuple{Int64, Int64})
	return length(childNodes(net, pos))
end

# function returning all nodes from `pos1` to `pos2` by assuming
# that the layer of `pos1` is lower or equal than `pos2`
function _connectingPath(net::AbstractNetwork, pos1::Tuple{Int64, Int64}, pos2::Tuple{Int64, Int64})
	l1,p1 = pos1
	l2,p2 = pos2
	
	@assert 0 < l1 ≤ l2
	@assert 0 < p1 ≤ n_tensors(net, l1)
	@assert 0 < l2 ≤ n_layers(net)
	@assert 0 < p2 ≤ n_tensors(net, l2)

	# first assume l1 ≤ l2, p1 < p2 for simplicity
	# later to the general thing
	# fast excit in case both are equal
	all(pos1 .== pos2) && (return Vector{Tuple{Int64, Int64}}())

	pos_cur = pos1 
	# first bring l1 and l2 to the same layer
	path_delta = Vector{Tuple{Int64, Int64}}(undef, l2 - l1)
	for jj in 1:(l2-l1)
		pos_cur = parentNode(net, pos_cur)
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
		posl = parentNode(net, posl)
		posr = parentNode(net, posr)
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

function connectingPath(net::AbstractNetwork, pos1::Tuple{Int64, Int64}, pos2::Tuple{Int64, Int64})
	if(pos1[1] < pos2[1])
		path = _connectingPath(net, pos1, pos2)
		return path[2:end]
	else
		path = _connectingPath(net, pos2, pos1)
		return reverse(path)[2:end]
	end	
end


function Base.iterate(net::AbstractNetwork)
	pos = (1, 1)
	return (pos, pos)
end

function Base.iterate(net::AbstractNetwork, state)
	#state == n_layers && return nothing
	state[1] == n_layers(net) && return nothing

	if state[2] == n_tensors(net, state[1])
		pos = (state[1] + 1, 1)
	else
		pos = (state[1], state[2] + 1)
	end
	return (pos, pos)
end

function Base.iterate(itr::Iterators.Reverse{<:AbstractNetwork})
	pos = (n_layers(itr.itr), 1)
	return (pos, pos)
end

function Base.iterate(itr::Iterators.Reverse{<:AbstractNetwork}, state)
	state == (1,1) && return nothing	
	
	if(state[2] == 1)
		n_l = state[1] - 1
		n_t = n_tensors(itr.itr, n_l)
		pos = (n_l, n_t)
	else
		pos = (state[1], state[2] - 1)
	end

	return (pos, pos)
end