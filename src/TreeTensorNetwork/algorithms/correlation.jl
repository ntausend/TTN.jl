# ITensor functionallity
function correlations(ttn::TreeTensorNetwork, op1, op2, pos::NTuple)
    pos_lin = linear_ind(physical_lattice(network(ttn)), pos)
    return correlations(ttn, op1, op2, pos_lin)
end
function correlations(ttn::TreeTensorNetwork, op1, op2, pos::Int)
    physlat = physical_lattice(network(ttn))
    res = map(eachindex(physlat)) do pp
        correlation(ttn, op1, op2, pos, pp)
    end
    dims = size(physlat)
    return reshape(res, dims)
end

function correlation(ttn::TreeTensorNetwork, op1, op2, pos1::NTuple, pos2::NTuple)
    pos1_lin = linear_lind(physical_lattice(network(ttn)), pos1)
    pos2_lin = linear_lind(physical_lattice(network(ttn)), pos2)
    return correlation(ttn, op1, op2, pos1_lin, pos2_lin)
end

function correlation(ttn::TreeTensorNetwork{N, ITensor}, op1::AbstractString, op2::AbstractString, pos1::Int, pos2::Int) where {N}
    if pos1 == pos2
        # fast exit using the expectation value
        op_new = "$op1 * $op2"
        return expect(ttn, op_new, pos1)
    end
    if pos1 < pos2
        return _correlation_pos1_le_pos2(ttn, op1, op2, pos1, pos2)
    else
        return conj(_correlation_pos1_le_pos2(ttn, op2, op1, pos2, pos1))
    end
end

# function for positions not being equal
# otherwise it is the expectation value of the product of operators
function _correlation_pos1_le_pos2(ttn::TreeTensorNetwork{N, ITensor}, op1::String, op2::String, pos1::Int, pos2::Int) where N
    @assert pos1 < pos2

    net = network(ttn)
    phys_sites = sites(ttn)
    # get the operators
    Opl = convert_cu(op(op1, phys_sites[pos1]), ttn[(1,1)])
    Opr = convert_cu(op(op2, phys_sites[pos2]), ttn[(1,1)])

    pos_parent1 = parent_node(net, (0,pos1))
    pos_parent2 = parent_node(net, (0,pos2))
    # get the minimal path connecting the two 
    path = vcat(pos_parent1, connecting_path(net, pos_parent1, pos_parent2))

    # finding the top node for the subtree
    _, topidx = findmax(first, path)
    top_pos = path[topidx]
    # split path into "left" and "right" part of the path
    # in general networks, they might not be equal
    
    path_l = path[1:topidx-1]
    path_r = path[end:-1:topidx+1]
    
    # move the orhtocenter to the top node for having all other subtrees collapse 
    ttnc = move_ortho!(copy(ttn), top_pos)
    # now calculate the flow of both operators to the end of the path
    for posl in path_l
        T = ttnc[posl]
        # getting the index shared by the tensor and
        # the current left_rg operator
        idx_shr = commonind(T, Opl)
        # now getting the link to the parent node. This link is always
        # labeled by the current layer number
        # no need for optimal_contraction_sequence here.. Opl and Opr only
        # operates on one leg
        idx_prnt = only(inds(T; tags = "nl=$(posl[1])"))
        Opl = (T * Opl) * dag(prime(T, idx_prnt, idx_shr))
    end
    for posr in path_r
        T = ttnc[posr]
        # getting the index shared by the tensor and
        # the current left_rg operator
        idx_shr = commonind(T, Opr)
        # now getting the link to the parent node. This link is always
        # labeled by the current layer number
        # no need for optimal_contraction_sequence here.. Opl and Opr only
        # operates on one leg
        idx_prnt = only(inds(T; tags = "nl=$(posr[1])"))
        Opr = (T * Opr) * dag(prime(T, idx_prnt, idx_shr)) 
    end

    T = ttnc[top_pos]

    idx_shrl = commonind(T, Opl)
    idx_shrr = commonind(T, Opr)
    # no need for optimal_contraction_sequence here.. Opl and Opr only
    # operates on one leg
    return ITensors.scalar(((T*Opl) * Opr)*dag(prime(T, idx_shrl, idx_shrr)))
end
