struct MPOWrapper{L, M, B} <: TTNKit.AbstractTensorProductOperator{L,B}
    lat::L
    data::M
    mapping::Vector{Int}
end


function MPOWrapper(lat::L, mpo::MPOHamiltonian, mapping::Vector{Int}) where L
    data = _wrapper_mpskit_mpo(mpo)

    # correct the first and last tensor
    tnfirst = data[1]
    tnlast  = data[end]
    codom   = codomain(tnfirst)[1]
    dom     =   domain(tnlast)[2]
    if sectortype(lat) == Trivial
        ctl = Tensor([jj == 1 ? 1 : 0 for jj in 1:dim(dom)], dom')
        ctr = Tensor([jj == dim(codom) ? 1 : 0  for jj in 1:dim(codom)], codom)
    else
        ctl = Tensor(zeros, codom')
        ctr = Tensor(zeros, dom)
        # these can only have the trivial rep
        trRep = keys(blocks(ctl))[1]
        blocks(ctl)[trRep][1] = 1
        trRep = keys(blocks(ctr))[1]
        blocks(ctr)[trRep][end] = 1
    end

    @plansor tmp1[-1,-2,-3] := ctl[1] * tnfirst[1, -1, -2, -3]
    @plansor tmp2[-1,-2;-3] := tnlast[-1, -2, -3, 1] * ctr[1]

    data[1] = TensorKit.permute(tmp1, (1,), (2,3))
    data[end] = tmp2


    return MPOWrapper{L, typeof(data), TensorKitBackend}(lat, data, mapping)
end

function _wrapper_mpskit_mpo(mpo::MPOHamiltonian)
    s = convert(SparseMPO,mpo.data)
    embeds = PeriodicArray(_embedders.([s[i].domspaces for i in 1:length(s)]))

    data = map(1:size(s,1)) do loc
        mapreduce(+,Iterators.product(1:s.odim,1:s.odim)) do (i,j)
            @plansor temp[-1 -2;-3 -4]:=embeds[loc][i][-1;1]*s[loc][i,j][1 -2;-3 2]*conj(embeds[loc+1][j][-4;2])
        end
    end)
    DenseMPO(data)
end
    end
    Vector{TensorMap}(data)
end

#TODO: Make this also for symmetry states
function _wrapper_itensors_mpo(ampo::OpSum, sites::Vector{<:Index{Int64}})
    ham = ITensors.MPO(ampo, sites)
    data = map(enumerate(ham)) do (i,h)
        dimHilb = ITensors.dim(inds(h)[end])

        if length(inds(h)) == 3
            dimLink = ITensors.dim(inds(h)[1])

            if i == 1
                TensorKit.permute(TensorKit.Tensor(array(h), (ℂ^dimLink)'*ℂ^dimHilb*(ℂ^dimHilb)'), (2,), (3,1))
            else
                TensorKit.permute(TensorKit.Tensor(array(h), ℂ^dimLink*ℂ^dimHilb*(ℂ^dimHilb)'), (1,2), (3,))
            end
        else
            dimLink1 = ITensors.dim(inds(h)[1])
            dimLink2 = ITensors.dim(inds(h)[2])
            TensorKit.permute(TensorKit.Tensor(ITensors.array(h), ℂ^dimLink1*(ℂ^dimLink2)'*ℂ^dimHilb*(ℂ^dimHilb)'), (1,3), (4,2))
        end
    end

    return Vector{TensorMap}(data)
end

function Hamiltonian(ampo::OpSum, type_str::AbstractString, lat::AbstractLattice{D, S, I, TensorKitBackend}; 
            mapping::Vector{Int} = collect(eachindex(lat)), kwargs...) where {D, S, I}
    sites = siteinds(type_str, number_of_sites(lat); kwargs...)
    data = _wrapper_itensors_mpo(ampo, sites)
    return MPOWrapper{typeof(lat), typeof(data), TensorKitBackend}(lat, data, mapping)
end

#ITensors constracters
# also include the mappings here

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
    return MPOWrapper{L, MPO, ITensorsBackend}(lat, mpoc, collect(eachindex(lat)))
end

function Hamiltonian(ampo::OpSum, lat::AbstractLattice{D, S, I, ITensorsBackend}) where{D, S, I}
    @assert isone(dimensionality(lat))
    @assert is_physical(lat)
    idx_lat = siteinds(lat)

    mpo = MPO(ampo, idx_lat)
    return MPOWrapper{typeof(lat), MPO, ITensorsBackend}(lat, mpo, collect(eachindex(lat)))
end
