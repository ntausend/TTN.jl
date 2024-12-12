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

include(joinpath(path_to_src, "j1j2_functions.jl"))

# symmetry broken phase
J1 = 1.0
J2 = 0.58
# dimensions of the lattice
dims = (2^2, 2^2)
# periodic boundary conditions
periodic = false
# if we want, we can use the gpu
conserve_qns = false
# performing 5 sweeps
number_of_sweeps = 5

# maximal bond dimension
maxdims = 10

net = BinaryRectangularNetwork(dims, "SpinHalf"; conserve_qns = conserve_qns)
lat = physical_lattice(net)

ttn0 = RandomTreeTensorNetwork(net; maxdim = nmaxdim, elT = Float64)
if use_gpu
    ttn0 = gpu(ttn0)
end

obs = EnergyObserver()

sp = dmrg(ttn0, tpo; number_of_sweeps = number_of_sweeps, maxdims, observer = obs)

ttn = sp.ttn
E = sp.current_energy
println("Final energy of the optimization is $(E)")

# can be plottet to see how the energy converges with the iterations
energy_convergence = obs.en_vec
println(energy_convergence)

# measurements of local observables
Xav = expect(ttn, "X")
Yav = expect(ttn, "Y")
Zav = expect(ttn, "Z")
println(Xav)
println(Yav)
println(Zav)

# if we want to save, we can use the HDF5 interface
if save
    h5open("results/wv_chi_$(maxdims).h5", "w") do f
        write(f, "wavefunction", cpu(ttn))
    end
end