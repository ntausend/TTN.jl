# todo include abstract model handling?
@with_kw struct TrivialModel
    en_offset::Float64 = 1
end

function Hamiltonian(md::TrivialModel, lat::AbstractLattice{1})
    sp = hilbertspace(node(lat,1))

    Id = isomorphism(sp, sp)
    ham = MPOHamiltonian(md.en_offset * Id)
    return MPO(lat, ham)
end