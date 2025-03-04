struct MPOWrapper{L, M} <: AbstractTensorProductOperator{L}
    lat::L
    data::M
    mapping::Vector{Int}
end

function Hamiltonian(mpo::MPO, lat::L; mapping::Vector{Int} = collect(eachindex(lat))) where{L}
    @assert is_physical(lat)
    @assert length(lat) == length(mpo)
    @assert isone(dimensionality(lat))
    idx_lat = siteinds(lat)

    mpoc = deepcopy(mpo)
    idx_mpo = first.(siteinds(mpoc,plev = 0))

    foreach(1:length(idx_lat)) do jj
        sj_lat = idx_lat[jj]
        sj_mpo = idx_mpo[jj]
        mpoc[jj] = replaceinds!(mpoc[jj], sj_mpo => sj_lat, prime(sj_mpo) => prime(sj_lat))
    end
    return MPOWrapper{L, MPO}(lat, mpoc, mapping)
end

"""
```julia
    Hamiltonian(ampo::OpSum, lat::L; mapping::Vector{Int} = collect(eachindex(lat))) where{L}
```

Creates an MPO used for the DMRG/TDVP simulations based on the abstract `OpSum` object of `ITensor`.
The mapping translates between the order of the tree tensor network and the spacial lattice enumeration.
The default is the standart one to one mapping. But in two and higher dimensions this might not be optimal and one have to choose a differnt mapping.
"""
function Hamiltonian(ampo::OpSum, lat::AbstractLattice; mapping = collect(eachindex(lat)))
    # @assert isone(dimensionality(lat))
    @assert is_physical(lat)
    # idx_lat = siteinds(lat)
    # idx_lat = map(mapping) do pos 
    idx_lat = map(inverse_mapping(mapping)) do pos 
        hilbertspace(lat[pos])
    end

    mpo = MPO(ampo, idx_lat)
    return MPOWrapper{typeof(lat), MPO}(lat, mpo, mapping)
end
