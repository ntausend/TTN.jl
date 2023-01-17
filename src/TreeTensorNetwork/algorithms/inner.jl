function inner(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork)
    net1 = network(ttn1)
    net2 = network(ttn2)

    dimensionality(net1) == dimensionality(net2) || return ComplexF64(0)
    physical_lattice(net1) == physical_lattice(net2) || return ComplexF64(0)

    return _inner(ttn1, ttn2)
end

function _inner(::TreeTensorNetwork{<:AbstractNetwork}, ::TreeTensorNetwork{<:AbstractNetwork})
    #, ::AbstractNetwork, ::TreeTensorNetwork, ::AbstractNetwork)
    error("Overlap function for not implemented for general lattices.")
end

function LinearAlgebra.norm(ttn::TreeTensorNetwork)
    oc = ortho_center(ttn)
    oc == (-1,-1) && return inner(ttn,ttn)
    return LinearAlgebra.norm(ttn[oc])
end

function LinearAlgebra.normalize!(ttn)
    oc = ortho_center(ttn)
    oc == (-1,-1) && return _reorthogonalize!(ttn, normalize = true)

    tn_n = ttn[oc]/LinearAlgebra.norm(ttn[oc])
    ttn[oc] = tn_n
    return ttn
end

# is this function valid for every network with same structure?
# so we can trop the <:BinaryNetwork restriction?
function _inner(ttn1::TreeTensorNetwork{N, T}, ttn2::TreeTensorNetwork{N, T}) where{N<:BinaryNetwork, T}

    net = network(ttn1)

    elT = promote_type(eltype(ttn1), eltype(ttn2))
    # check in case if symmetric the Top node for qn correspondence
    if T == TensorMap && !(sectortype(net) == Trivial)
        dom1 = domain(ttn1[number_of_layers(net),1])
        dom2 = domain(ttn2[number_of_layers(net),1])
        dom1 == dom2 || return zero(elT)
    elseif T isa ITensor && !(sectortype(net) == Int64)
        fl1 = flux(ttn1[number_of_layers(net), 1])
        fl2 = flux(ttn2[number_of_layers(net), 2])
        fl1 == fl2 || return zero(elT)
    end

    # contruct the network starting from the first layer upwards
    #ns = number_of_sites(net)
    
    phys_lat = physical_lattice(net)
    if T == TensorMap
        res = map(phys_lat) do (nd)
            isomorphism(hilbertspace(nd), hilbertspace(nd))
        end
    else
        res = map(phys_lat) do nd
            delta(hilbertspace(nd), prime(hilbertspace(nd)))
        end
    end


    for ll in eachlayer(net)
        nt = number_of_tensors(net,ll)
        res_new = Vector{T}(undef, nt)
        for pp in eachindex(net, ll)
            childs_idx = getindex.(child_nodes(net, (ll,pp)),2)
            tn1 = ttn1[ll,pp]
            tn2 = ttn2[ll,pp]
            rpre1 = res[childs_idx[1]]
            rpre2 = res[childs_idx[2]]
            res_new[pp] = _dot_inner(tn1, tn2, rpre1, rpre2)
        end
        res = res_new
    end
    # better exception
    length(res) == 1 || error("Tree Tensor Contraction don't leed to a single resulting tensor.")
    res = res[1]
    if T == TensorMap
        space_dom = domain(res)
        space_co  = codomain(res)
        dim_tk(space_dom) == 1   || error("Tree Tensor Contraction don't leed to a tensor with one dimensional domain ")
        dim_tk(space_co)  == 1   || error("Tree Tensor Contraction don't leed to a tensor with one dimensional codomain ")
        @tensor sres = res[1,1]
    else
        sres = ITensors.scalar(res)
    end

    return sres
end


function _dot_inner(tn1::AbstractTensorMap, tn2::AbstractTensorMap, rpre1::AbstractTensorMap, rpre2::AbstractTensorMap)
    tmp = @tensor tmp[-1; -2] := conj(tn1[r1, r2, -1])*rpre1[r1, k1]*rpre2[r2,k2]*tn2[k1,k2,-2]
    return tmp
end
function _dot_inner(tn1::ITensor, tn2::ITensor, rpre1::ITensor, rpre2::ITensor)
    return dag(prime(tn1))*((tn2 * rpre1) * rpre2)
end