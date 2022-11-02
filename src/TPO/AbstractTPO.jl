abstract type AbstractTensorProductOperator{L <: AbstractLattice} end

dimensionality(::AbstractTensorProductOperator{N}) where N = dimensionality(N)
lattice(tpo::AbstractTensorProductOperator) = tpo.lat


struct MPO <: TTNKit.AbstractTensorProductOperator{TTNKit.SimpleLattice{1}}
    lat::TTNKit.SimpleLattice{1}
    data::Vector{TensorMap}
end


function transverseIsingHamiltonian(parameters::Tuple, lat::SimpleLattice)

    J, g = parameters

    hilb = TTNKit.hilbertspace(node(lat, 1))

    id = [1 0; 0 1]
    σ_z = [1 0; 0 -1]
    σ_x = [0 1; 1 0]

    matrix = zeros(dim(hilb), dim(hilb), 3, 3)
    matrix[:,:,1,1] .= matrix[:,:,3,3] .= id
    matrix[:,:,2,1] .= σ_z
    matrix[:,:,3,1] .= g*σ_x
    matrix[:,:,3,2] .= J*σ_z

    hamiltonian = TensorKit.permute(Tensor(matrix, hilb*(hilb)'*ℂ^3*(ℂ^3)'),(1,3),(2,4))

    return MPO(lat, Vector{TensorMap}( fill(hamiltonian, number_of_sites(lat)) ))
end