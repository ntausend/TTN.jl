# this is needed if you have tiny project dependent environments
#using Pkg

# this has to be changed if you use encap to '../../'
#rootpath = "./"

#Pkg.activate(rootpaht)

# if you run this for the first time... necessary to have the dependencies
#Pkg.instantiate()

using ITensors
using TTNKit
using TTNKit: ProductTreeTensorNetwork, Hamiltonian, physical_lattice
using JLD2

# expands the TTNKit toolkit by the trenary trees
#include("TrenaryTreeDefinition.jl")
# should be included...

# gets the hamiltonian
include("hamiltonian.jl")
# get the main function
include("main.jl")
include("jld2_interface.jl")


n_sweeps = 10
maxdim   = 10
conserve_qns = false
use_random_init = true


# can be vector or number, vector applies the noise at j-th position in the j-th sweep. If j>length(noise), it applies
# the last value for the rest of the sweeps, only needed in the beginning of the sweep to go out of the product state.
# alternative: use a random state with full bond dimension. Only a good way if no QN's are envolved, or the QN space
# is finite as in the ℤ₂ of the Ising model
if use_random_init
        noise = 0.0
else
        noise = [1E-2, 1E-2, 1E-2, 1E-2, 0.0]
end

# subspace expansion for increasing of bond dimension. It's value is in percentage of the full two site hilbertspace
p = 0.2

h = 1
pbc = false

N = 9

savename = "ttn_trenary_h$(h)_N$(N).jld2"

mdl_params = (N = N, h = h, pbc = pbc)



ψ = main(n_sweeps::Int64, maxdim::Int64; mdl_params, noise, p, conserve_qns, use_random_init)


write_tree_jld2(savename, ψ)
