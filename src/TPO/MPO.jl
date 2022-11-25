struct MPO{L<:AbstractLattice{1}} <: TTNKit.AbstractTensorProductOperator{L}
    lat::L
    data::DenseMPO
end


function MPO(lat::L, mpo::MPOHamiltonian) where L
    data = _wrapper_mpskit_mpo(mpo)
    return MPO{L}(lat, data)
end

function _wrapper_mpskit_mpo(mpo::MPOHamiltonian)
    s = convert(SparseMPO,mpo)
    embeds = PeriodicArray(_embedders.([s[i].domspaces for i in 1:length(s)]))

    data = PeriodicArray(map(1:size(s,1)) do loc
        mapreduce(+,Iterators.product(1:s.odim,1:s.odim)) do (i,j)
            @plansor temp[-1 -2;-3 -4]:=embeds[loc][i][-1;1]*s[loc][i,j][1 -2;-3 2]*conj(embeds[loc+1][j][-4;2])
        end
    end)
    DenseMPO(data)
end