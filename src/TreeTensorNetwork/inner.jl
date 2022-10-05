function inner(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork)
    net1 = network(ttn1)
    net2 = network(ttn2)

    dimensionality(net1) == dimensionality(net2) || return ComplexF64(0)
    physicalLattice(net1) == physicalLattice(net2) || return ComplexF64(0)

    return _inner(ttn1, net1, ttn2, net2)
end

function _inner(::TreeTensorNetwork, ::AbstractNetwork, ::TreeTensorNetwork, ::AbstractNetwork)
    error("Overlap function for not implemented for general lattices.")
end

function _inner(ttn1::TreeTensorNetwork, net1::BinaryNetwork, 
                ttn2::TreeTensorNetwork, ::BinaryNetwork)


    # check in case if symmetric the Top node for qn correspondence
    if !(sectortype(net1) == Trivial)
        dom1 = domain(ttn1[n_layers(net1),1])
        dom2 = domain(ttn2[n_layers(net1),1])
        dom1 == dom2 || return zero(ComplexF64)
    end

    # contruct the network starting from the first layer upwards
    #ns = number_of_sites(net1)
    
    phys_lat = physicalLattice(net1)
    res = map(phys_lat) do (nd)
        isomorphism(hilbertspace(nd), hilbertspace(nd))
    end


    for ll in eachlayer(net1)
        nt = n_tensors(net1,ll)
        res_new = Vector{TensorMap}(undef, nt)
        for pp in eachsite(net1, ll)
            childs_idx = getindex.(childNodes(net1, (ll,pp)),2)
            tn1 = ttn1[ll,pp]
            tn2 = ttn2[ll,pp]
            rpre1 = res[childs_idx[1]]
            rpre2 = res[childs_idx[2]]

            @tensor tmp[-1; -2] := conj(tn1[r1, r2, -1])*rpre1[r1, k1]*rpre2[r2,k2]*tn2[k1,k2,-2]
            res_new[pp] = tmp
        end
        res = res_new
    end
    length(res) == 1 || error("Tree Tensor Contraction don't leed to a single resulting tensor.")
    res = res[1]
    space_dom = domain(res)
    space_co  = codomain(res)
    dim(space_dom) == 1   || error("Tree Tensor Contraction don't leed to a tensor with one dimensional domain ")
    dim(space_co)  == 1   || error("Tree Tensor Contraction don't leed to a tensor with one dimensional codomain ")

    @tensor sres = res[1,1]
    return sres
end