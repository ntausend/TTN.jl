
# todo include abstract model handling?
@with_kw struct TransverseFieldIsing
    #boundary_conditions <: BC # somehow encode if periodic or open ?
    J::Float64 = 1 # NN Interaction strength
    g::Float64 # Transverse Field interaction
end

function Hamiltonian(md::TransverseFieldIsing, lat::AbstractLattice; mapping::Vector{Int} = collect(TTNKit.eachindex(lat)))
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