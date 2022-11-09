abstract type AbstractTensorProductOperator{L <: AbstractLattice} end

dimensionality(::AbstractTensorProductOperator{N}) where N = dimensionality(N)
lattice(tpo::AbstractTensorProductOperator) = tpo.lat


struct MPO <: TTNKit.AbstractTensorProductOperator{TTNKit.SimpleLattice{1}}
    lat::TTNKit.SimpleLattice{1}
    data::Vector{AbstractTensorMap}
end


function transverseIsingHamiltonian(parameters::Tuple, lat::SimpleLattice)

    J, g = parameters

    hilb = TTNKit.hilbertspace(node(lat, 1))

    id = [1 0; 0 1]
    σ_z = [1 0; 0 -1]
    σ_x = [0 1; 1 0]

    hamiltonian = zeros(dim(hilb), dim(hilb), 3, 3)
    hamiltonian[:,:,1,1] .= hamiltonian[:,:,3,3] .= id
    hamiltonian[:,:,2,1] .= σ_z
    hamiltonian[:,:,3,1] .= g*σ_x
    hamiltonian[:,:,3,2] .= J*σ_z

    return MPO(lat, Vector{TensorMap}( fill(Tensor(hamiltonian, (hilb)'*hilb*ℂ^3*(ℂ^3)'), number_of_sites(lat)) ))
end