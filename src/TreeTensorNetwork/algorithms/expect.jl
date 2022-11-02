# expectation value of a onsite operator i.e.: <n_j>

function expect(ttn::TreeTensorNetwork{D}, op::OnSiteOperator) where{D}
    physlat = physical_lattice(network(ttn))
    res = map(eachindex(physlat)) do (pos)
        expect(ttn, op, pos)
    end
    dims = size(physlat)
    return reshape(res, dims)
end

function expect(ttn::TreeTensorNetwork{D}, op::OnSiteOperator, pos::Union{NTuple{D,Int},Int}) where{D}
    net = network(ttn)
    return _expect(ttn, net, op, pos)
end

function _expect(ttn::TreeTensorNetwork{D}, net::AbstractNetwork{D}, op::OnSiteOperator, pos::NTuple{D,Int}) where{D}
    return _expect(ttn,net, op, linear_ind(physical_lattice(net),pos))
end

function _expect(ttn::TreeTensorNetwork{D}, net::AbstractNetwork{D}, op::OnSiteOperator, pos::Int) where{D}

    ttnc = copy(ttn)
    physlat = physical_lattice(net)
    hilbttn = hilbertspace(node(physlat,1))

    doo1  = domain(op)
    codo1 = codomain(op)
    if !(ProductSpace(hilbttn) == doo1 == codo1)
        error("Codomain and domain of operator $op not matching with local hilbertspace.")
    end

    # linear position in the D-dimensional lattice
    ch_pos = (0,pos)
    # finding parent node position

    parent_pos = parent_node(net, ch_pos)

    # move ortho_center to parent pos
    move_ortho!(ttnc, parent_pos)

    # find the index of the child
    idx_ch = index_of_child(net, ch_pos)
    
    tnc = ttnc[parent_pos]

    # splitting index to have the child index alone in the codomain
    idx_codom, idx_dom = split_index(net, parent_pos, idx_ch)
    # permute the indices 
    tnc = TensorKit.permute(tnc, idx_codom, idx_dom)
    # perform the contraction
    res = dot(tnc, op*tnc)

    return real(res)
end