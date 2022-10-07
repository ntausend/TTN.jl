function inner(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork)
    net1 = network(ttn1)
    net2 = network(ttn2)

    dimensionality(net1) == dimensionality(net2) || return ComplexF64(0)
    physical_lattice(net1) == physical_lattice(net2) || return ComplexF64(0)

    return _inner(ttn1, net1, ttn2, net2)
end

function _inner(::TreeTensorNetwork, ::AbstractNetwork, ::TreeTensorNetwork, ::AbstractNetwork)
    error("Overlap function for not implemented for general lattices.")
end

function TensorKit.norm(ttn::TreeTensorNetwork)
    oc = ortho_center(ttn)
    oc == (-1,-1) && return inner(ttn,ttn)
    return TensorKit.norm(ttn[oc])
end

function TensorKit.normalize!(ttn)
    oc = ortho_center(ttn)
    oc == (-1,-1) && return _reorthogonalize!(ttn, normalize = true)

    tn_n = ttn[oc]/TensorKit.norm(ttn[oc])
    ttn[oc] = tn_n
    return ttn
end

# is this function valid for every network with same structure?
# so we can trop the <:BinaryNetwork restriction?
function _inner(ttn1::TreeTensorNetwork, net::N, 
                ttn2::TreeTensorNetwork, ::N) where{N<:BinaryNetwork}

    elT = promote_type(eltype(ttn1), eltype(ttn2))
    # check in case if symmetric the Top node for qn correspondence
    if !(sectortype(net) == Trivial)
        dom1 = domain(ttn1[number_of_layers(net),1])
        dom2 = domain(ttn2[number_of_layers(net),1])
        dom1 == dom2 || return zero(elT)
    end

    # contruct the network starting from the first layer upwards
    #ns = number_of_sites(net)
    
    phys_lat = physical_lattice(net)
    res = map(phys_lat) do (nd)
        isomorphism(hilbertspace(nd), hilbertspace(nd))
    end


    for ll in eachlayer(net)
        nt = number_of_tensors(net,ll)
        res_new = Vector{TensorMap}(undef, nt)
        for pp in eachindex(net, ll)
            childs_idx = getindex.(child_nodes(net, (ll,pp)),2)
            tn1 = ttn1[ll,pp]
            tn2 = ttn2[ll,pp]
            rpre1 = res[childs_idx[1]]
            rpre2 = res[childs_idx[2]]

            @tensor tmp[-1; -2] := conj(tn1[r1, r2, -1])*rpre1[r1, k1]*rpre2[r2,k2]*tn2[k1,k2,-2]
            res_new[pp] = tmp
        end
        res = res_new
    end
    # better exception
    length(res) == 1 || error("Tree Tensor Contraction don't leed to a single resulting tensor.")
    res = res[1]
    space_dom = domain(res)
    space_co  = codomain(res)
    dim(space_dom) == 1   || error("Tree Tensor Contraction don't leed to a tensor with one dimensional domain ")
    dim(space_co)  == 1   || error("Tree Tensor Contraction don't leed to a tensor with one dimensional codomain ")

    @tensor sres = res[1,1]
    return sres
end