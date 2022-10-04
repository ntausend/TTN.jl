function inner(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork)
    net1 = network(ttn1)
    net2 = network(ttn2)

    dimensionality(net1) == dimensionality(net2) || return ComplexF64(0)
    lattice(net1) == lattice(net2) || return ComplexF64(0)

    return _inner(ttn1, net1, ttn2, net2)
end

function inner(::TreeTensorNetwork, ::AbstractLattice, ::TreeTensorNetwork, ::AbstractLattice)
    error("Overlap function for not implemented for general lattices.")
end

function _inner(ttn1::TreeTensorNetwork, net1::BinaryNetwork, 
                ttn2::TreeTensorNetwork, net2::BinaryNetwork)

    nl = n_layers(net1)
    # contruct the network starting from the first layer upwards
    ns = number_of_sites(lattice(net1))
    
    res = map(1:ns) do (jj)
        isomorphism(hilbertspace(node(lattice(net1), jj)), hilbertspace(node(lattice(net2), jj)))
    end

    for ll in 1:nl
        nt = n_tensors(net1,ll)
        res_new = Vector{TensorMap}(undef, nt)
        for pp in 1:nt
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
    space_dom == space_co || error("Tree Tensor Contraction dont't leed to a tensor with same domain and codomain")
    dim(space_dom) == 1   || error("Tree Tensor Contraction don't leed to a tensor with one dimensional domain and codomain")
    
    @tensor sres = res[1,1]
    return sres
end