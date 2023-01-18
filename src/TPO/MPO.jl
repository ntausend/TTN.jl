struct MPOWrapper{L<:AbstractLattice{1},M, B} <: TTNKit.AbstractTensorProductOperator{L,B}
    lat::L
    data::M
end


function MPOWrapper(lat::L, mpo::MPOHamiltonian) where L
    data = _wrapper_mpskit_mpo(mpo)
    return MPOWrapper{L, typeof(data), TensorKitBackend}(lat, data)
end

function _wrapper_mpskit_mpo(mpo::MPOHamiltonian)
    s = convert(SparseMPO,mpo.data)
    embeds = PeriodicArray(_embedders.([s[i].domspaces for i in 1:length(s)]))

    data = PeriodicArray(map(1:size(s,1)) do loc
        mapreduce(+,Iterators.product(1:s.odim,1:s.odim)) do (i,j)
            @plansor temp[-1 -2;-3 -4]:=embeds[loc][i][-1;1]*s[loc][i,j][1 -2;-3 2]*conj(embeds[loc+1][j][-4;2])
        end
    end)
    DenseMPO(data)
end

function Hamiltonian(mpo::MPO, lat::L) where{L}
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
    return MPOWrapper{L, MPO, ITensorsBackend}(lat, mpoc)
end

function Hamiltonian(ampo::OpSum, lat::L) where{L}
    @assert isone(dimensionality(lat))
    @assert is_physical(lat)
    idx_lat = siteinds(lat)

    mpo = MPO(ampo, idx_lat)
    return MPOWrapper{L, MPO, ITensorsBackend}(lat, mpo)
end