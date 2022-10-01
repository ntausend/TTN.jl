# special struct for one dimensional binary networks with some simplifications
struct OneDimensionalBinaryNetwork <: AbstractNetwork{1}
	n_layer::Vector{Int64}
	bonddims::Vector{Int64}
	lattice::BinaryChain
	function OneDimensionalBinaryNetwork(n::Int64, bonddims::Vector{Int64}, local_dim::Int64)
		n_layer = [2^(n - jj) for jj in 1:n]

		#local_inds = collect(1:n_layer[1])
		return new(n_layer, bonddims, BinaryChain(2^n, local_dim))
	end			
end


# exact formular for this kind of network
n_tensors(net::OneDimensionalBinaryNetwork) = 2^(n_layers(net)) - 1

# since we do not save them per default, they need to be created
# sepeartly
function adjacencyMatrix(net::OneDimensionalBinaryNetwork, l::Int64)
	n_this = n_tensors(net, l)
	n_next = n_tensors(net, l+1)
	pos_this = collect(1:n_this)
	I = repeat(collect(1:n_next), 2)
	J = vcat(pos_this[1:2:end], pos_this[2:2:end])
	return sparse(I,J,repeat([1], n_this), n_next, n_this)
end

# direct formular for this kind of network
function childNodes(net::OneDimensionalBinaryNetwork, pos::Tuple{Int64,Int64})
	l, p = pos
	@assert 0 < l ≤ n_layers(net)
	@assert 0 < p ≤ n_tensors(net, l)

	#l == 1 && (return nothing)
	return [(l - 1, 2*p - 1), (l - 1, 2*p)]
end

n_childNodes(net::OneDimensionalBinaryNetwork, pos::Tuple{Int64, Int64}) = 2

function parentNode(net::OneDimensionalBinaryNetwork, pos::Tuple{Int64,Int64})
	l,p = pos
	
	@assert 0 < l ≤ n_layers(net)
	@assert 0 < p ≤ n_tensors(net, l)
	
	l == n_layers(net) && (return nothing)
	return (l+1, (p+1)÷2)
end	


function index_of_child(net::OneDimensionalBinaryNetwork, pos_child)
	return mod1(pos_child[2],2)
end