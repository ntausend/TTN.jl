
# todo include abstract model handling?
@with_kw struct TransverseFieldIsing
    #boundary_conditions <: BC # somehow encode if periodic or open ?
    J::Float64 = 1 # NN Interaction strength
    g::Float64 # Transverse Field interaction
end

function Hamiltonian_dep(md::TransverseFieldIsing, lat::AbstractLattice{1}; mapping::Vector{Int} = collect(eachindex(lat)))
    J = md.J
    g = md.g
    
    if sectortype(lat) == Trivial
        sx = [0 1; 1  0]
        sz = [1 0; 0 -1]

        sp = ComplexSpace(size(sx, 1))

        Sx = TensorMap(sx, sp, sp)
        Sz = TensorMap(sz, sp, sp)


        ham1 = @mpoham(sum(J * Sx{i}*Sx{j} for (i,j) in nearest_neighbours(lat, mapping)))
        ham2 = @mpoham(sum(g * Sz{k} for k in eachindex(lat)))
        ham = ham1 + ham2
    else
        sp = Rep[ℤ₂](0 => 1, 1 => 1)

        sz = TensorMap(zeros, ComplexF64, sp, sp)
        blocks(sz)[ℤ₂(1)] .= 1
        blocks(sz)[ℤ₂(0)] .= -1

        sx = TensorMap(ones, ComplexF64, sp * Rep[ℤ₂](1 => 1), sp)
        @tensor nn[-1 -2; -3 -4] := sx[-1 1; -3] * conj(sx[-4 1; -2])

        ham = MPOHamiltonian(J * nn) + MPOHamiltonian(g * sz)
    end
    return MPOWrapper(lat, ham, mapping)
end

function Hamiltonian(md::TransverseFieldIsing, lat::AbstractLattice{D,S,I, ITensorsBackend};
        mapping::Vector{Int} = collect(TTNKit.eachindex(lat))) where{D,S,I}
    J = md.J
    g = md.g
    
    ampo = OpSum();

    for i in eachindex(lat)
        ITensors.add!(ampo, g, "Z",i)
    end

    for (i,j) in nearest_neighbours(lat, mapping)
        ITensors.add!(ampo, J, "X",i,"X",j)
    end

    return Hamiltonian(ampo, lat; mapping = mapping);
end

function Hamiltonian(md::TransverseFieldIsing, lat::AbstractLattice{D,S,I, TensorKitBackend};
        mapping::Vector{Int} = collect(TTNKit.eachindex(lat))) where{D,S,I}
    J = md.J
    g = md.g
    
    ampo = OpSum();

    for i in eachindex(lat)
        ITensors.add!(ampo, g, "Z",i)
    end

    for (i,j) in nearest_neighbours(lat, mapping)
        ITensors.add!(ampo, J, "X",i,"X",j)
    end

    return Hamiltonian(ampo, "SpinHalf", lat; mapping = mapping);
end
