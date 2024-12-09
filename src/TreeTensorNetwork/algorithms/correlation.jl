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
    pos1_lin = linear_ind(physical_lattice(network(ttn)), pos1)
    pos2_lin = linear_ind(physical_lattice(network(ttn)), pos2)
    return correlation(ttn, op1, op2, pos1_lin, pos2_lin)
end

function correlation(ttn::TreeTensorNetwork, op1::AbstractString, op2::AbstractString, pos1::Int, pos2::Int)
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
function _correlation_pos1_le_pos2(ttn::TreeTensorNetwork, op1::String, op2::String, pos1::Int, pos2::Int)
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


### general n point correlations ###
function correlation(ttn::TreeTensorNetwork, ops::Vector{String}, pos::Vector{Tuple{Int,Int}})
    pos_lin = [linear_ind(physical_lattice(network(ttn)), posi) for posi in pos]
    return correlation(ttn, ops, pos_lin)
end

function correlation(ttn::TreeTensorNetwork, ops::Vector{String}, pos::Vector{Int})
    net = network(ttn)
    phys_sites = sites(ttn)

    # find the top position
    pos_parent = [parent_node(net, (0,posi)) for posi in pos]
    paths = [vcat(pos_parent1, connecting_path(net, pos_parent1, pos_parent2)) for (pos_parent1, pos_parent2) in Iterators.product(pos_parent, pos_parent) if pos_parent1 != pos_parent2]

    # finding the top node for the subtree
    top_path_pos_idx = [findmax(first, path)[2] for path in paths]
    top_path_pos = [path[top_path_pos_i] for (path,top_path_pos_i) in zip(paths, top_path_pos_idx)]
    _,top_pos_idx = findmax(first, top_path_pos)
    top_pos = top_path_pos[top_pos_idx]

    ttnc = move_ortho!(copy(ttn), top_pos)

    ops_pos = [(convert_cu(op(opsi, phys_sites[posi]), ttnc[(1,1)]), (0,posi)) for (opsi,posi) in zip(ops,pos)]

    for ll in 1:top_pos[1]-1
      temp_ops_pos = []

      for pp in eachindex(net,ll)
        idx = findall(x -> parent_node(net, x[2]) == (ll,pp), ops_pos)
        isempty(idx) && continue

        T = ttnc[(ll,pp)]
        temp_ops = [ops_pos[i][1] for i in idx]

        idx_shr = [commonind(T, temp_opsi) for temp_opsi in temp_ops]
        idx_prnt = only(inds(T; tags = "nl=$(ll)"))
        append!(temp_ops_pos, [(reduce(*, temp_ops, init = T) * dag(prime(T, idx_prnt, idx_shr...)), (ll,pp))])
      end

      ops_pos = temp_ops_pos

    end

    T = ttnc[top_pos]
    idx = findall(x -> parent_node(net, x[2]) == top_pos, ops_pos)
    temp_ops = [ops_pos[i][1] for i in idx]
    idx_shr = [commonind(T, temp_opsi) for temp_opsi in temp_ops]

    return ITensors.scalar(reduce(*, temp_ops, init = T) * dag(prime(T, idx_shr...)))
end;
