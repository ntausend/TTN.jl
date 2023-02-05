
# function correlation(ttn::TreeTensorNetwork, op1::AbstractTensorMap, op2::AbstractTensorMap, 
#                      pos1::Union{NTuple{D,Int}, Int}, pos2::Union{NTuple{D,Int}, Int}) where{D}
#     net = network(ttn)
#     return _correlation(ttn, net, op1, op2, pos1, pos2)
# end
#
# function _correlation(ttn::TreeTensorNetwork, net::AbstractNetwork, op1::AbstractTensorMap, op2::AbstractTensorMap, 
#                      pos1::NTuple{D,Int}, pos2::NTuple{D,Int}) where{D}
#     pos1_lin = linear_ind(physical_lattice(net), pos1)
#     pos2_lin = linear_ind(physical_lattice(net), pos2)
#     return _correlation(ttn, net, op1, op2, pos1_lin, pos2_lin)
# end
#
# function _correlation(::TreeTensorNetwork, ::AbstractNetwork, ::AbstractTensorMap, ::AbstractTensorMap, ::Int, ::Int)
#     error("Not implemented for general networks.... TODO")
# end
#
# # currently only onsite operators supported, i.e. neutral operators which can be represented as a matrix
# function _correlation_same_parent(ttn::TreeTensorNetwork, net::AbstractNetwork,
#                      op1::OnSiteOperator{S}, op2::OnSiteOperator{S}, 
#                      parent_pos::Tuple{Int,Int}, idx_ch1::Int, idx_ch2::Int) where{S}
#     ttnc = copy(ttn)
#     # move ortho_center to the parent position
#     move_ortho!(ttnc, parent_pos)
#     # now start contracting the network
#     T = ttnc[ortho_center(ttnc)]
#
#     # construct the new codomain as the tow child indices
#
#
#     idx_codomain = (idx_ch1, idx_ch2)
#     idx_domain   = deleteat!(collect(1:number_of_child_nodes(net, parent_pos)+1), idx_codomain)
#
#     @tensor ocomp[-1,-2;-3,-4] := op1[-1,-3]*op2[-2,-4]
#
#     T = TensorKit.permute(T, idx_codomain, Tuple(idx_domain))
#
#     return dot(T,ocomp*T)
# end

function _correlation(ttn::TreeTensorNetwork{N,ITensor,ITensorsBackend}, op1::String, op2::String, pos1::Int, pos2::Int) where N
    net = network(ttn)
    phys_sites = sites(ttn)
    Opl = convert_cu(op(op1, phys_sites[pos1]), ttn[(1,1)])
    Opr = convert_cu(op(op2, phys_sites[pos2]), ttn[(1,1)])

    # getting the parent nodes
    pos_parent1 = parent_node(net, (0,pos1))
    pos_parent2 = parent_node(net, (0,pos2))
    
    if pos1 == pos2
        ttnc = copy(ttn)
        move_ortho!(ttnc, parent_node(net, (0,pos1)))

        T = ttnc[pos_parent1]
        T_adj = dag(ttnc[pos_parent1])
        commonInd = commonind(T_adj, Opl)
        Opl = prime(Opl)
        replaceind!(T_adj, commonInd, commonInd'')

        opt_seq = ITensors.optimal_contraction_sequence(Opl, Opr, T, T_adj)
        return real(array(contract(Opl, Opr, T, T_adj; sequence = opt_seq))[1])
    end

    path = vcat(pos_parent1, connecting_path(net, pos_parent1, pos_parent2)...)
    # split path into "left" and "right" part of equal length
    (_,topidx) = findmax(first, path)
    topPos = path[topidx]
    path_l = path[1:topidx-1]
    path_r = path[end:-1:topidx+1]

    ttnc = copy(ttn)
    move_ortho!(ttnc, topPos)

    # contract operators with tensors along the left and right path respectively, indices are changed appropriately
    for (posl, posr) in zip(path_l, path_r)
        Tl = ttnc[posl]
        Tl_adj = dag(ttnc[posl])
        commonIndl = commonind(Tl_adj, Opl)
        topIndl = inds(Tl_adj)[end]
        replaceinds!(Tl_adj, [commonIndl, topIndl], [commonIndl', topIndl''])

        Tr = ttnc[posr]
        Tr_adj = dag(ttnc[posr])
        commonIndr = commonind(Tr_adj, Opr)
        topIndr = inds(Tr_adj)[end]
        replaceinds!(Tr_adj, [commonIndr, topIndr], [commonIndr', topIndr''])

        opt_seql = ITensors.optimal_contraction_sequence(Opl, Tl, Tl_adj)
        opt_seqr = ITensors.optimal_contraction_sequence(Opr, Tr, Tr_adj)
        Opl = replaceprime(contract(Opl, Tl, Tl_adj; sequence = opt_seql), 2 => 1)
        Opr = replaceprime(contract(Opr, Tr, Tr_adj; sequence = opt_seqr), 2 => 1)
    end

    # contract both branches with the top node of the path
    Ttop = ttnc[topPos]
    Ttop_adj = prime(dag(ttnc[topPos]))

    if topPos != (number_of_layers(net),1)
        topInd = inds(Ttop_adj)[end]
        replaceind!(Ttop_adj, topInd, noprime(topInd))
    end

    opt_seq = ITensors.optimal_contraction_sequence(Opl, Opr, Ttop, Ttop_adj)
    return real(array(contract(Opl, Opr, Ttop, Ttop_adj; sequence = opt_seq))[1])
end

# function _correlation(ttn::TreeTensorNetwork, net::BinaryNetwork, op1::OnSiteOperator{S}, op2::OnSiteOperator{S}, 
#                      pos1::Int, pos2::Int) where{S}
#     if pos1 == pos2 
#         @tensor ocomp[-1;-2] := op1[-1,s]*op2[s,-2]
#         return expect(ttn, ocomp, pos1)
#     end
#     
#     physlat = physical_lattice(net)
#     hilbttn = hilbertspace(node(physlat,1))
#
#     for op in (op1, op2)
#         doo1  = domain(op)
#         codo1 = codomain(op)
#         if !(ProductSpace(hilbttn) == doo1 == codo1)
#             error("Codomain and domain of operator $op not matching with local hilbertspace.")
#         end
#     end
#
#     # linear index inside the network
#     ch_pos1 = (0,pos1)
#     ch_pos2 = (0,pos2)
#
#     # index of childs 
#     idx_ch1 = index_of_child(net, ch_pos1)
#     idx_ch2 = index_of_child(net, ch_pos2)
#
#
#     # getting the parent nodes
#     pos_parent1 = parent_node(net, ch_pos1)
#     pos_parent2 = parent_node(net, ch_pos2)
#     if (pos_parent1 == pos_parent2)
#         return _correlation_same_parent(ttn, net, op1, op2, pos_parent1, idx_ch1, idx_ch2)
#     end
#     # path connecting the two parent nodes
#     # along this path we need to contract the network
#     path = connecting_path(net, pos_parent1, pos_parent2)
#
#
#     ttnc = copy(ttn)
#     # move ortho_center to the parent position
#     move_ortho!(ttnc, pos_parent1)
#     # now start contracting the network
#     L = ttnc[ortho_center(ttnc)]
#
#     # reorganize the parent tensor T to be contracted with the operator Occ
#     idx_codom, idx_dom = split_index(net, pos_parent1, idx_ch1)
#     perm = vcat(idx_codom..., idx_dom...)
#     L_p = TensorKit.permute(L, idx_codom, idx_dom)
#     L_p = TensorKit.permute(op1*L_p, Tuple(perm[1:end-1]), Tuple(perm[end]))
#     L = adjoint(L)*L_p
#
#     # now construct the closing Tensor with the final operator insertion
#     # for this, push the child node attached to the operator to the codomain
#     idx_codom, idx_dom = split_index(net, pos_parent2, idx_ch2)
#     perm = vcat(idx_codom..., idx_dom...)
#     R_p = TensorKit.permute(ttnc[pos_parent2], idx_codom, idx_dom)
#     # contract the operator insertion
#     # and permute the indices back to the original order
#     R_p = TensorKit.permute(op2*R_p, Tuple(perm[1:end-1]), Tuple(perm[end]))
#     # now contract the adjoint enviromnet to R_p
#     R = adjoint(ttnc[pos_parent2])*R_p
#
#     # finding the index of the top node
#     idx_top = findmax(x -> x[1], path)[2]
#     path_l = path[1:idx_top-1]
#     path_r = Iterators.reverse(path[idx_top+1:end-1])
#
#     # now buliding the left enviromnet, everything goes towards parent nodes
#     pos_prev = pos_parent1
#     for pos in path_l
#         T_next = ttnc[pos]
#         # current node has to be child of top node, so first find the correct index
#         idx_ch = index_of_child(net, pos_prev)
#         # index splitting
#         idx_codom, idx_dom = split_index(net, pos, idx_ch)
#         perm = vcat(idx_codom..., idx_dom...)
#         T_next_p = TensorKit.permute(T_next, idx_codom, idx_dom)
#         L = TensorKit.permute(L*T_next_p, Tuple(perm[1:end-1]), Tuple(perm[end]))
#         L = adjoint(T_next)*L
#         pos_prev = pos
#     end
#     pos_l_end = pos_prev
#
#     # now build the right enviromnet, everything goes towards parent nodes
#     pos_prev = pos_parent2
#     for pos in path_r
#         T_next = ttnc[pos]
#     
#         idx_ch = index_of_child(net, pos_prev)
#         # index splitting
#         idx_codom, idx_dom = split_index(net, pos, idx_ch)
#         perm = vcat(idx_codom..., idx_dom...)
#         T_next_p = TensorKit.permute(T_next, idx_codom, idx_dom)
#         R = TensorKit.permute(R*T_next_p, Tuple(perm[1:end-1]), Tuple(perm[end]))
#         R = adjoint(T_next)*R
#         pos_prev = pos
#     end
#     pos_r_end = pos_prev
#
#     pos_top = path[idx_top]
#     T_top = ttnc[pos_top]
#     # getting idx of left child
#     idx_chl = index_of_child(net, pos_l_end)
#     #index split
#     idx_codom, idx_dom = split_index(net, pos_top, idx_chl)
#     perm = vcat(idx_codom..., idx_dom...)
#     T_top_p = TensorKit.permute(T_top, idx_codom, idx_dom)
#     
#     T_top = TensorKit.permute(L*T_top_p, Tuple(perm[1:end-1]), Tuple(perm[end]))
#     # getting idx of right child
#     idx_chr = index_of_child(net, pos_r_end)
#     idx_codom, idx_dom = split_index(net, pos_top, idx_chr)
#     perm = vcat(idx_codom..., idx_dom...)
#     T_top_p = TensorKit.permute(T_top, idx_codom, idx_dom)
#     T_top = TensorKit.permute(R*T_top_p, Tuple(perm[1:end-1]), Tuple(perm[end]))
#
#     res = dot(ttnc[pos_top], T_top)
#     return res
# end
