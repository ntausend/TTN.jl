# expectation value of a onsite operator i.e.: <n_j>

function expect(ttn::TreeTensorNetwork, op)
    physlat = physical_lattice(network(ttn))
    res = map(eachindex(physlat)) do (pos)
        expect(ttn, op, pos)
    end
    dims = size(physlat)
    return reshape(res, dims)
end
function expect(ttn::TreeTensorNetwork, op, pos::NTuple)
    return expect(ttn, op, linear_ind(physical_lattice(network(ttn)),pos))
end

function expect(ttn::TreeTensorNetwork{L,TensorMap}, O::TensorMap, pos::Int) where{L}

    net = network(ttn)
    ttnc = copy(ttn)
    physlat = physical_lattice(net)
    hilbttn = hilbertspace(node(physlat,1))

    doo1  = domain(O)
    codo1 = codomain(O)
    if !(ProductSpace(hilbttn) == doo1 == codo1)
        error("Codomain and domain of operator $O not matching with local hilbertspace.")
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
    res = dot(tnc, O*tnc)

    return res
end

function expect(ttn::TreeTensorNetwork{L,ITensor}, op_str::AbstractString, pos::Int) where{L}

    net = network(ttn)
    ttnc = copy(ttn)

    idx = siteinds(net)[pos]
    O   = TTNKit.convert_cu(op(op_str, idx), ttn[(1,1)])

    # linear position in the D-dimensional lattice
    ch_pos = (0,pos)
    # finding parent node position

    parent_pos = parent_node(net, ch_pos)

    # move ortho_center to parent pos
    move_ortho!(ttnc, parent_pos)

    # find the index of the child
    T = ttnc[parent_pos]
    # perform the contraction
    res = dot(T, noprime(O*T))

    return res
end
