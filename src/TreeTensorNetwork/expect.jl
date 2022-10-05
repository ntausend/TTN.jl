function expect(ttn::TreeTensorNetwork{D}, op::TensorMap) where{D}
    physlat = physicalLattice(network(ttn))
    res = map(eachsite(physlat)) do (pos)
        expect(ttn, op, pos)
    end
    dims = size(physlat)
    return reshape(res, dims)
end



function expect(ttn::TreeTensorNetwork{D}, op::TensorMap, pos::Union{NTuple{D,Int},Int}) where{D}
    net = network(ttn)
    return _expect(ttn, net, op, pos)
end

function _expect(ttn::TreeTensorNetwork{D}, net::AbstractNetwork{D}, op::TensorMap, pos::NTuple{D,Int}) where{D}
    return _expect(ttn,net, op, to_linear_ind(physicalLattice(net),pos))
end

function _expect(::TreeTensorNetwork, ::AbstractNetwork,::TensorMap,::Int)
    error("Overlap function for not implemented for general lattices.")
end


function _expect(ttn::TreeTensorNetwork{D}, net::BinaryNetwork{D}, op::TensorMap, pos::Int) where{D}

    ttnc = copy(ttn)
    physlat = physicalLattice(net)
    hilbttn = hilbertspace(node(physlat,1))

    doo1  = domain(op)
    codo1 = codomain(op)
    if !(ProductSpace(hilbttn) == doo1 == codo1)
        error("Codomain and domain of operator $op not matching with local hilbertspace.")
    end

    # linear position in the D-dimensional lattice
    ch_pos = (0,pos)
    # finding parent node position

    parent_pos = parentNode(net, ch_pos)

    # move ortho_center to parent pos
    move_ortho!(ttnc, parent_pos)

    # find the index of the child
    idx_ch = index_of_child(net, ch_pos)
    
    tnc = ttnc[parent_pos]

    if (idx_ch == 1)
        @tensor res = conj(tnc[s, 1, 2]) * op[s,s′] * tnc[s′, 1, 2]
    else
        @tensor res = conj(tnc[1, s, 2]) * op[s,s′] * tnc[1, s′, 2]
    end
    
    return real(res)
    # This might somehow work for general geometries...
    # now preform a permutation of the legs to have the domain being the one to calculate the
    # expectation value
    #=
    allinds = 1:(1 + length(codomain(tnc)))
    res_inds = deleteat!(collect(allinds), idx_ch)

    tnc = TensorKit.permute(tnc, (idx_ch,), Tuple(res_inds))
    res = adjoint(tnc)*(op*tnc)
    #@tensor res = res[1,1]
    #tnc = permute()
    return res
    =#

end