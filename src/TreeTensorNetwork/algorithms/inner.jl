"""
```julia
   inner(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork)
```

Calculates the overlapp between the two tensor networks `ttn1` and `ttn2`.
"""
function inner(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork)
    net1 = network(ttn1)
    net2 = network(ttn2)

    dimensionality(net1) == dimensionality(net2) || return ComplexF64(0)
    physical_lattice(net1) == physical_lattice(net2) || return ComplexF64(0)

    return _inner(ttn1, ttn2)
end

# for calculating the inner product when ttn1 is being optimized at ortho_center
function inner(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork, update_center::Tuple{Int,Int})
    net1 = network(ttn1)
    net2 = network(ttn2)

    dimensionality(net1) == dimensionality(net2) || return ComplexF64(0)
    physical_lattice(net1) == physical_lattice(net2) || return ComplexF64(0)

    return _inner(ttn1, ttn2, update_center)
end

function inner(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork, T::ITensor, update_site::Tuple{Int,Int})
    net1 = network(ttn1)
    net2 = network(ttn2)

    dimensionality(net1) == dimensionality(net2) || return ComplexF64(0)
    physical_lattice(net1) == physical_lattice(net2) || return ComplexF64(0)

    return _inner(ttn1, ttn2, T, update_site)
end

function _inner(::TreeTensorNetwork{<:AbstractNetwork}, ::TreeTensorNetwork{<:AbstractNetwork})
    #, ::AbstractNetwork, ::TreeTensorNetwork, ::AbstractNetwork)
    error("Overlap function for not implemented for general lattices.")
end

function LinearAlgebra.norm(ttn::TreeTensorNetwork)
    oc = ortho_center(ttn)
    oc == (-1,-1) && return sqrt(abs(inner(ttn,ttn)))
    return LinearAlgebra.norm(ttn[oc])
end

function LinearAlgebra.normalize!(ttn)
    oc = ortho_center(ttn)
    oc == (-1,-1) && return _reorthogonalize!(ttn, normalize = true)

    tn_n = ttn[oc]/LinearAlgebra.norm(ttn[oc])
    ttn[oc] = tn_n
    return ttn
end

# for calculating the inner product when ttn1 is being optimized at ortho_center
function _inner(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork, update_center::Tuple{Int,Int})

    net = network(ttn1)

    if ortho_center(ttn1) != update_center
        move_ortho!(ttn1, update_center)
        #error("The ortho_center of the TTN to be updated is not the same as the update_center.")
    end

    # move the ortho_center of ground state TTN to the update_center
    if ortho_center(ttn2) != update_center
        move_ortho!(ttn2, update_center)
        #error("The ortho_center of the ground state TTN is not the same as the update_center.")
    end

    phys_lat = physical_lattice(net)
    res = map(phys_lat) do nd
        [delta(dag.(hilbertspace(nd)), prime(hilbertspace(nd)))]
    end

    # do normal contraction until the the layer before the update_center
    for ll in 1:update_center[1]-1
        res_new = [Vector{ITensor}(undef,1) for _ in 1:number_of_tensors(net,ll)]#Vector{ITensor}(undef, number_of_tensors(net,ll))
        for pp in eachindex(net,ll)
            childs_idx = getindex.(child_nodes(net, (ll,pp)),2)
            tn1 = ttn1[ll,pp]
            tn2 = ttn2[ll,pp]
            rpre1 = res[childs_idx[1]][1]
            rpre2 = res[childs_idx[2]][1]
            res_new[pp][1] = _dot_inner(tn1, tn2, rpre1, rpre2)
        end
        res = res_new
    end

    #println("Got up to the update_center layer")

    res_new = [Vector{ITensor}(undef,1) for _ in 1:number_of_tensors(net,update_center[1])]#Vector{ITensor}(undef, number_of_tensors(net,ll))
    for pp in eachindex(net,update_center[1])
        childs_idx = getindex.(child_nodes(net, (update_center[1],pp)),2)
        tn1 = ttn1[update_center[1],pp]
        tn2 = ttn2[update_center[1],pp]
        rpre1 = res[childs_idx[1]][1]
        rpre2 = res[childs_idx[2]][1]
        if pp == update_center[2]
            res_new[pp] = [dag(prime(tn1)),((tn2 * rpre1) * rpre2)]
        else
            res_new[pp][1] = _dot_inner(tn1, tn2, rpre1, rpre2)
        end
    end
    res = res_new

    #println("Finished the update_center layer")

    for ll in update_center[1]+1:number_of_layers(net)
        #println("Working on layer: ", ll)
        res_new = [Vector{ITensor}(undef,1) for _ in 1:number_of_tensors(net,ll)]#Vector{ITensor}(undef, number_of_tensors(net,ll))
        for pp in eachindex(net,ll)
            #println("Working on position: ", pp)
            childs_idx = getindex.(child_nodes(net, (ll,pp)),2)
            tn1 = ttn1[ll,pp]
            tn2 = ttn2[ll,pp]
            if length(res[childs_idx[1]]) == 1
                rpre1 = res[childs_idx[1]][1]
                if length(res[childs_idx[2]]) == 1
                    rpre2 = res[childs_idx[2]][1]
                    res_new[pp][1] = _dot_inner(tn1, tn2, rpre1, rpre2)
                else
                    rpre2 = res[childs_idx[2]]
                    other_contraction = (dag(prime(tn1)) * (tn2 * rpre1)) * rpre2[2]
                    res_new[pp] = [rpre2[1], other_contraction]
                end
            else
                rpre1 = res[childs_idx[1]]
                rpre2 = res[childs_idx[2]][1]
                other_contraction = (dag(prime(tn1)) * (tn2 * rpre2)) * rpre1[2]
                res_new[pp] = [rpre1[1], other_contraction]
            end
        end
        res = res_new
    end

    #display(res)

    return res[1][2]
end

# is this function valid for every network with same structure?
# so we can trop the <:BinaryNetwork restriction?
function _inner(ttn1::TreeTensorNetwork{N, T}, ttn2::TreeTensorNetwork{N, T}) where{N<:BinaryNetwork, T}

    net = network(ttn1)

    elT = promote_type(eltype(ttn1), eltype(ttn2))
    # check in case if symmetric the Top node for qn correspondence
    #if T == TensorMap && !(sectortype(net) == Trivial)
        #dom1 = domain(ttn1[number_of_layers(net),1])
        #dom2 = domain(ttn2[number_of_layers(net),1])
        #dom1 == dom2 || return zero(elT)
    #elseif T isa ITensor && !(sectortype(net) == Int64)
    if !(sectortype(net) == Int64)
        fl1 = flux(ttn1[number_of_layers(net), 1])
        fl2 = flux(ttn2[number_of_layers(net), 2])
        fl1 == fl2 || return zero(elT)
    end

    # contruct the network starting from the first layer upwards
    #ns = number_of_sites(net)
    
    phys_lat = physical_lattice(net)
    #if T == TensorMap
    #    res = map(phys_lat) do (nd)
    #        isomorphism(hilbertspace(nd), hilbertspace(nd))
    #    end
    #else
    res = map(phys_lat) do nd
        delta(hilbertspace(nd), prime(hilbertspace(nd)))
    end
    #end


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
    #if T == TensorMap
    #    space_dom = domain(res)
    #    space_co  = codomain(res)
    #    dim_tk(space_dom) == 1   || error("Tree Tensor Contraction don't leed to a tensor with one dimensional domain ")
    #    dim_tk(space_co)  == 1   || error("Tree Tensor Contraction don't leed to a tensor with one dimensional codomain ")
    #    @tensor sres = res[1,1]
    #else
    sres = ITensors.scalar(res)
    #end

    return sres
end

# full overlap where ttn2 replaces the internal tensor with the given tensor T at position pos
function _inner(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork, T::ITensor, update_site::Tuple{Int,Int})

    net = network(ttn1)

    #top_pos = (TTNKit.number_of_layers(net),1)
    #move_ortho!(ttn1, top_pos)
    #move_ortho!(ttn2, top_pos)

    # contruct the network starting from the first layer upwards
    #ns = number_of_sites(net)
    
    phys_lat = physical_lattice(net)
    if T == TensorMap
        res = map(phys_lat) do (nd)
            isomorphism(hilbertspace(nd), hilbertspace(nd))
        end
    else
        res = map(phys_lat) do nd
            delta(dag.(hilbertspace(nd)), prime(hilbertspace(nd)))
        end
    end


    for ll in eachlayer(net)
        nt = number_of_tensors(net,ll)
        res_new = Vector{ITensor}(undef, nt)
        for pp in eachindex(net, ll)
            childs_idx = getindex.(child_nodes(net, (ll,pp)),2)
            tn1 = ttn1[ll,pp]
            tn2 = (ll,pp) == update_site ? T : ttn2[ll,pp]
            rpre1 = res[childs_idx[1]]
            rpre2 = res[childs_idx[2]]
            res_new[pp] = _dot_inner(tn1, tn2, rpre1, rpre2)
            #println("At layer $ll and position $pp with result ", res_new[pp])
        end
        res = res_new
    end
    # better exception
    length(res) == 1 || error("Tree Tensor Contraction don't leed to a single resulting tensor.")
    res = res[1]
    sres = ITensors.scalar(res)

    return sres
end

function _dot_inner(tn1::ITensor, tn2::ITensor, rpre1::ITensor, rpre2::ITensor)
    return dag(prime(tn1))*((tn2 * rpre1) * rpre2)
end