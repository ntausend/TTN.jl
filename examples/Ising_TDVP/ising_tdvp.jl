# This is an example for minimizing the energy of the J1J2 Model on a square lattice.
using ITensors
using TTN
using ITensorMPS: AbstractObserver, measure!, OpSum, expect
using HDF5

use_gpu = false
if use_gpu
    using CUDA
end
save = false

include("./ising_functions.jl")


dims = (2^2,2^2)
typ = ComplexF64

# couplings
J = -1
g = -0.5
h = 0
periodic = false

#parameters for time evolution
dt = 1e-1 #timestep
t0 = 0. #initial time
tmax = 1. #final time
D = 64 #maximal bond dimension

ind = ITensors.siteinds("S=1/2",prod(dims))
net = TTN.BinaryNetwork(dims, ind)
lat = TTN.physical_lattice(net)

# transversely polarised initial state
states = fill("Up", prod(dims))

# initialize ttn as a product state
ttn = TTN.ProductTreeTensorNetwork(net, states)
# increase the bond dimension to set value D by padding with zeros
ttn = TTN.increase_dim_tree_tensor_network_zeros(ttn, maxdim = D)
# send to gpu if needed
if use_gpu
  ttn = TTN.gpu(ttn)
end

obs = tdvpObserver()


# setup tpo
tpo = TPO(TFIMHam(J, g, h, lat), lat)

tdvp_finish = TTN.tdvp(ttn, tpo, initialtime = t0, finaltime = tmax, timestep = dt, observer = obs)

# obs is a DataFrame containing all measurements, can be saved to disk if needed
if save
  name_obs = "results/obs_J_$(J)_g_$(g)_h_$(h)_chi_$(D)"
  savedata(name_obs, obs)
end
