
function correlation(ttn::TreeTensorNetwork{D}, op1::AbstractTensorMap, op2::AbstractTensorMap, 
                     pos1::Union{NTuple{D,Int}, Int}, pos2::Union{NTuple{D,Int}, Int}) where{D}
    net = network(ttn)
    return _correlation(ttn, net, op1, op2, pos1, pos2)
end

function _correlation(ttn::TreeTensorNetwork{D}, net::AbstractNetwork{D}, op1::AbstractTensorMap, op2::AbstractTensorMap, 
                     pos1::NTuple{D,Int}, pos2::NTuple{D,Int}) where{D}
    pos1_lin = linear_ind(physical_lattice(net), pos1)
    pos2_lin = linear_ind(physical_lattice(net), pos2)
    return _correlation(ttn, net, op1, op2, pos1_lin, pos2_lin)
end

function _correlation(::TreeTensorNetwork{D}, ::AbstractNetwork{D}, ::AbstractTensorMap, ::AbstractTensorMap, ::Int, ::Int) where{D}
    error("Not implemented for general networks.... TODO")
end

# currently only onsite operators supported, i.e. neutral operators which can be represented as a matrix
function _correlation_same_parent(ttn::TreeTensorNetwork{D}, net::AbstractNetwork{D},
                     op1::OnSiteOperator{S}, op2::OnSiteOperator{S}, 
                     parent_pos::Tuple{Int,Int}, idx_ch1::Int, idx_ch2::Int) where{D, S}
    ttnc = copy(ttn)
    # move ortho_center to the parent position
    move_ortho!(ttnc, parent_pos)
    # now start contracting the network
    T = ttnc[ortho_center(ttnc)]

    # construct the new codomain as the tow child indices


    idx_codomain = (idx_ch1, idx_ch2)
    idx_domain   = deleteat!(collect(1:number_of_child_nodes(net, parent_pos)+1), idx_codomain)

    @tensor ocomp[-1,-2;-3,-4] := op1[-1,-3]*op2[-2,-4]

    T = TensorKit.permute(T, idx_codomain, Tuple(idx_domain))

    return dot(T,ocomp*T)
end

function _correlation(ttn::TreeTensorNetwork{D}, net::BinaryNetwork{D}, op1::OnSiteOperator{S}, op2::OnSiteOperator{S}, 
                     pos1::Int, pos2::Int) where{D, S}
    if pos1 == pos2 
        @tensor ocomp[-1;-2] := op1[-1,s]*op2[s,-2]
        return _expect(ttn, net, ocomp, pos1)
    end
    
    physlat = physical_lattice(net)
    hilbttn = hilbertspace(node(physlat,1))

    for op in (op1, op2)
        doo1  = domain(op)
        codo1 = codomain(op)
        if !(ProductSpace(hilbttn) == doo1 == codo1)
            error("Codomain and domain of operator $op not matching with local hilbertspace.")
        end
    end

    # linear index inside the network
    ch_pos1 = (0,pos1)
    ch_pos2 = (0,pos2)

    # index of childs 
    idx_ch1 = index_of_child(net, ch_pos1)
    idx_ch2 = index_of_child(net, ch_pos2)


    # getting the parent nodes
    pos_parent1 = parent_node(net, ch_pos1)
    pos_parent2 = parent_node(net, ch_pos2)
    if (pos_parent1 == pos_parent2)
        return _correlation_same_parent(ttn, net, op1, op2, pos_parent1, idx_ch1, idx_ch2)
    end
    # path connecting the two parent nodes
    # along this path we need to contract the network
    path = connecting_path(net, pos_parent1, pos_parent2)


    ttnc = copy(ttn)
    # move ortho_center to the parent position
    move_ortho!(ttnc, pos_parent1)
    # now start contracting the network
    L = ttnc[ortho_center(ttnc)]

    # reorganize the parent tensor T to be contracted with the operator Occ
    idx_codom, idx_dom = split_index(net, pos_parent1, idx_ch1)
    perm = vcat(idx_codom..., idx_dom...)
    L_p = TensorKit.permute(L, idx_codom, idx_dom)
    L_p = TensorKit.permute(op1*L_p, Tuple(perm[1:end-1]), Tuple(perm[end]))
    L = adjoint(L)*L_p

    # now construct the closing Tensor with the final operator insertion
    # for this, push the child node attached to the operator to the codomain
    idx_codom, idx_dom = split_index(net, pos_parent2, idx_ch2)
    perm = vcat(idx_codom..., idx_dom...)
    R_p = TensorKit.permute(ttnc[pos_parent2], idx_codom, idx_dom)
    # contract the operator insertion
    # and permute the indices back to the original order
    R_p = TensorKit.permute(op2*R_p, Tuple(perm[1:end-1]), Tuple(perm[end]))
    # now contract the adjoint enviromnet to R_p
    R = adjoint(ttnc[pos_parent2])*R_p

    # finding the index of the top node
    idx_top = findmax(x -> x[1], path)[2]
    path_l = path[1:idx_top-1]
    path_r = Iterators.reverse(path[idx_top+1:end-1])

    # now buliding the left enviromnet, everything goes towards parent nodes
    pos_prev = pos_parent1
    for pos in path_l
        T_next = ttnc[pos]
        # current node has to be child of top node, so first find the correct index
        idx_ch = index_of_child(net, pos_prev)
        # index splitting
        idx_codom, idx_dom = split_index(net, pos, idx_ch)
        perm = vcat(idx_codom..., idx_dom...)
        T_next_p = TensorKit.permute(T_next, idx_codom, idx_dom)
        L = TensorKit.permute(L*T_next_p, Tuple(perm[1:end-1]), Tuple(perm[end]))
        L = adjoint(T_next)*L
        pos_prev = pos
    end
    pos_l_end = pos_prev

    # now build the right enviromnet, everything goes towards parent nodes
    pos_prev = pos_parent2
    for pos in path_r
        T_next = ttnc[pos]
    
        idx_ch = index_of_child(net, pos_prev)
        # index splitting
        idx_codom, idx_dom = split_index(net, pos, idx_ch)
        perm = vcat(idx_codom..., idx_dom...)
        T_next_p = TensorKit.permute(T_next, idx_codom, idx_dom)
        R = TensorKit.permute(R*T_next_p, Tuple(perm[1:end-1]), Tuple(perm[end]))
        R = adjoint(T_next)*R
        pos_prev = pos
    end
    pos_r_end = pos_prev

    pos_top = path[idx_top]
    T_top = ttnc[pos_top]
    # getting idx of left child
    idx_chl = index_of_child(net, pos_l_end)
    #index split
    idx_codom, idx_dom = split_index(net, pos_top, idx_chl)
    perm = vcat(idx_codom..., idx_dom...)
    T_top_p = TensorKit.permute(T_top, idx_codom, idx_dom)
    
    T_top = TensorKit.permute(L*T_top_p, Tuple(perm[1:end-1]), Tuple(perm[end]))
    # getting idx of right child
    idx_chr = index_of_child(net, pos_r_end)
    idx_codom, idx_dom = split_index(net, pos_top, idx_chr)
    perm = vcat(idx_codom..., idx_dom...)
    T_top_p = TensorKit.permute(T_top, idx_codom, idx_dom)
    T_top = TensorKit.permute(R*T_top_p, Tuple(perm[1:end-1]), Tuple(perm[end]))

    res = dot(ttnc[pos_top], T_top)
    return res
end