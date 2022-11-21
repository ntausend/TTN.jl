using TTNKit
using TensorKit
using Plots

# observables
function x_pol(ttn::TreeTensorNetwork)
    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)
    σ_x = TensorMap([0 1; 1 0], ℂ^2 ← ℂ^2)   
    
    return sum(TTNKit.expect(ttn, σ_x))/len
end

function xx_pol(ttn::TreeTensorNetwork, distance::Integer)
    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)
    σ_x = TensorMap([0 1; 1 0], ℂ^2 ← ℂ^2)   

    correlations = [TTNKit.correlation(ttn, σ_x, σ_x, pp, pp+distance) for pp in 1:len-distance]
    
    return sum(correlations)/len
end

function z_pol(ttn::TreeTensorNetwork)
    len = TTNKit.number_of_sites(TTNKit.network(ttn))
    σ_z = TensorMap([1 0; 0 -1], ℂ^2 ← ℂ^2) 
    
    return sum(TTNKit.expect(ttn, σ_z))/len
end

function zz_pol(ttn::TreeTensorNetwork, distance::Integer)
    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)
    σ_z = TensorMap([1 0; 0 -1], ℂ^2 ← ℂ^2) 

    correlations = [TTNKit.correlation(ttn, σ_z, σ_z, pp, pp+distance) for pp in 1:len-distance]
    
    return sum(correlations)/len
end


function entanglementEntropy(ttn::TreeTensorNetwork)
    net = TTNKit.network(ttn)
    pos = (TTNKit.number_of_layers(net),1)
    U, S, V, eps = TensorKit.tsvd(ttn[pos], (1,), (2,3))
    println(S)
    return 0 
end

# parameters
n_layers = 3
maxBondDim = 10

t = 0.
dt = 1e-2
tmax = 4.

net = BinaryChainNetwork(n_layers, TTNKit.SpinHalfNode)

# "right" polarized state
states = fill("Right", TTNKit.number_of_sites(net))

# "up" polarized state
# states = fill("Up", TTNKit.number_of_sites(net))

# initialize tree tensor network and increase bond dimension to maxBondDim
ttn = ProductTreeTensorNetwork(net, states) 
ttn = increase_dim_tree_tensor_network_randn(ttn, maxdim = maxBondDim, factor = 10e-12)

# initialize tensor product operator and calculate initial environments
tpo = transverseIsingHamiltonian((-1., -2.), TTNKit.physical_lattice(net))
ptpo = ProjTensorProductOperator(ttn, tpo)

# run TDVP algorithm
tdvprun(ttn, ptpo, dt, tmax)
