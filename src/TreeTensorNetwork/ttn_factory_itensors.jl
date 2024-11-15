function _initialize_empty_ttn(net::AbstractNetwork)
    ttn = Vector{Vector{ITensor}}(undef, number_of_layers(net))
    foreach(eachlayer(net)) do ll
        ttn[ll] = Vector{ITensor}(undef, number_of_tensors(net, ll))
    end
    return ttn
end

function _construct_product_tree_tensor_network(net::AbstractNetwork, states::Vector{<:AbstractString}, _elT::DataType)
    if length(states) != length(physical_lattice(net))
        throw(DimensionMismatch("Number of physical sites and and initial vals don't match"))
    end

    ttn = _initialize_empty_ttn(net)
    state_vecs = map(zp -> state(zp[1], zp[2]), zip(physical_lattice(net), states))
    indices = Vector{Vector{Index}}(undef, length(ttn))
    

    indices[1] = Vector{Index}(undef, length(ttn[1]))
    for pp in eachindex(net, 1)

        chnds = child_nodes(net, (1, pp))
        state_vec = map(n -> state_vecs[n[2]], chnds)
        if hasqns(state_vec[1])
            upflux = mapreduce(flux, +, state_vec)
            uplink = dag(Index(upflux => 1; tags = "Link,nl=1,np=$pp"))
        else
            uplink = Index(1; tags = "Link,nl=1,np=$pp")
        end
        indices[1][pp] = uplink
        
        ttn[1][pp] = reduce(*, state_vec; init = state(uplink, 1))
        #ttn[1][pp] = state_vec[1] * state_vec[2] * state(uplink,1)
    end

    elT = promote_type(eltype(ttn[1][1]), _elT)
    ttn[1][:] = one(elT) .* ttn[1][:]

    for ll in Iterators.drop(eachlayer(net),1)
        indices[ll] = Vector{Index}(undef, length(ttn[ll]))
        for pp in eachindex(net, ll)
            chnds = child_nodes(net, (ll,pp))
            codom_vec = map(chnds) do (_,pc)
                dag(indices[ll-1][pc])
            end
            states_chd = map(idx -> state(idx,1), codom_vec)
            if hasqns(states_chd[1])
                upflux = mapreduce(flux, +, states_chd)
                uplink = dag(Index(upflux => 1; tags = "Link,nl=$ll,np=$pp"))
            else
                uplink = Index(1; tags = "Link,nl=$ll,np=$pp")
            end
            indices[ll][pp] = uplink
            ttn[ll][pp] = reduce(*, states_chd, init = one(elT)*state(uplink,1))
        end
    end
    # now we need to correct the last tensor to have no head index
    ttn[end][end] = ttn[end][end] * state(dag(indices[end][end]),1)
    return ttn
end


function _construct_random_tree_tensor_network(net::AbstractNetwork, maxdim::Int, elT::DataType)
    ttn = _initialize_empty_ttn(net)
    domains, codomains = _build_domains_and_codomains(net,  maxdim)
    # we need to correct the last domain
    uplink = dag(combinedind(combiner(codomains[end][1]...)))
    uplink = settags(uplink,tags(domains[end][1]))
    #fused_last = fuse(codomains[end][1])
    # last domain should simply be one dimensional fixing the network
    domains[end][1] = _correct_domain(uplink, 1)

    for (ll,pp) in NodeIterator(net)
        dom  = domains[ll][pp]
        codom = codomains[ll][pp]
        ttn[ll][pp] = random_itensor(elT, codom..., dom)
    end

    # correct the top node legs
    ttn[end][1] = ttn[end][1] * state(dag(domains[end][1]), 1)

    return ttn
end

function _construct_random_tree_tensor_network(net::AbstractNetwork, target_charge, maxdim::Int, elT::DataType, tries::Int)
    ttn = _initialize_empty_ttn(net)
    domains, codomains = _build_domains_and_codomains(net, target_charge, maxdim, tries)

    for (ll,pp) in NodeIterator(net)
        dom  = domains[ll][pp]
        codom = codomains[ll][pp]
        ttn[ll][pp] = random_itensor(elT, codom..., dom)
    end

    # correct the top node legs
    ttn[end][1] = ttn[end][1] * state(dag(domains[end][1]), 1)
    return ttn
end

function _correct_domain(domain_new::Index{I}, maxdim::Int) where{I}
    # getting current dimension of domain
    dim_total = dim(domain_new)

    # we are below the maxdim, so fast return
    dim_total ≤ maxdim && return domain_new

    I == Int64 && return Index(maxdim; tags = tags(domain_new))

    qn_vec = space(domain_new)
    # get the different sectors
    sects  = map(first,qn_vec)
    # get the dimensionality of the sectors
    dimsec = map(last, qn_vec)
    # now build a multinomial distribution wich draws maxdim samples
    ps = dimsec ./ dim_total

    rng = Multinomial(maxdim, ps)
    dims_new = rand(rng)
    # get all 0 irreps
    idx_zero = findall(iszero, dims_new)
    if !isnothing(idx_zero) 
        dims_new = deleteat!(dims_new, idx_zero)
        sects    = deleteat!(sects, idx_zero)
    end
    new_irreps = map(sd -> Pair(sd...), zip(sects, dims_new))

    return Index(new_irreps; tags = tags(domain_new), dir = dir(domain_new))
end

function _build_domains_and_codomains(net::AbstractNetwork, maxdim::Int)
    phys_lat = physical_lattice(net)

    codomains = Vector{Vector{Vector{Index}}}(undef, number_of_layers(net))
    domains = Vector{Vector{Index}}(undef, number_of_layers(net))
    foreach(eachlayer(net)) do ll
        codomains[ll] = Vector{Vector{Index}}(undef, number_of_tensors(net, ll))
        domains[ll]   = Vector{Index}(undef, number_of_tensors(net, ll)) 
    end
    indices_cur = map(x -> hilbertspace(node(phys_lat, x)), eachindex(phys_lat))

    for ll in eachlayer(net)
        for pp in eachindex(net, ll)
            chd = child_nodes(net, (ll,pp))
            indices_ch = map(n -> indices_cur[n[2]], chd)
            # getting the fused index
            uplink = dag(combinedind(combiner(indices_ch...)))
            uplink = settags(uplink,"Link,nl=$ll,np=$pp")
            # correct the the dimension to have maximal dimension of maxdim
            uplink = _correct_domain(uplink, maxdim)
            domains[ll][pp] = uplink
            codomains[ll][pp] = indices_ch
        end
        indices_cur = dag.(domains[ll])
    end
    return domains, codomains
end

function _build_domains_and_codomains(net::AbstractNetwork{<:AbstractLattice{D,S,I}}, target_charge, maxdim::Int, tries::Int) where{D,S, I}
    if(I == Int64 || I == Nothing)
        target_sp = Index(1; tags = "Link,nl=$(number_of_layers(net)),np=1")
    else
        target_sp = Index(target_charge => 1, tags = "Link,nl=$(number_of_layers(net)),np=1")
    end

    codomains = Vector{Vector{Vector{Index}}}(undef, number_of_layers(net))
    domains = Vector{Vector{Index}}(undef, number_of_layers(net))
    foreach(eachlayer(net)) do ll
        codomains[ll] = Vector{Vector{Index}}(undef, number_of_tensors(net, ll))
        domains[ll]   = Vector{Index}(undef, number_of_tensors(net, ll)) 
    end

    for tt in 1:tries

        domains, codomains = _build_domains_and_codomains(net, maxdim)
        # correct the last domain
        domains[end][end] = target_sp

        I == Int64 || I == Nothing &&  break
        #fusing the last codomains and see if target_space is contained
        uplink = dag(combinedind(combiner(codomains[end][1]...)))
        sects1 = map(first, space(uplink))
        sects2 = map(first, space(target_sp))
        isempty(intersect(sects1, sects2)) ||  break

        if(tt == tries)
            error("Maximal numbers of tries reached but no valid domain/codomain pattern was found for targetcharge $(I)")
        end
    end
    return domains, codomains
end



#===========================================================================================
                Enlarging TTN Tensors by some subspace expansion methods to
                have full simulation space. Only supported for trivial tensors!!
===========================================================================================#


function increase_dim_tree_tensor_network_zeros(ttn::TreeTensorNetwork; maxdim::Int = 1,
    orthogonalize::Bool = true, normalize::Bool = orthogonalize, _elT = eltype(ttn))

    net = network(ttn)
    elT = promote_type(_elT, eltype(ttn))

    # if (!sectortype(net) == Trivial)
    #     throw(ArgumentError("Increasing subspace by filling with zeros only allowed without Quantum Numbers"))
    # end
    
    ttn_new = _initialize_empty_ttn(net)

    domains, codomains = _build_domains_and_codomains(net, maxdim)

    for (ll,pp) in NodeIterator(net)
        dom  = domains[ll][pp]
        codom = codomains[ll][pp]
        data_temp = (ll == number_of_layers(net)) ? zeros(elT, ITensors.dim.((codom...,))) : zeros(elT, ITensors.dim.((codom..., dom))) 

        pos_it = Iterators.product(UnitRange.(1, ITensors.dim.(inds(ttn[ll,pp])))...)
        for pos in pos_it
          data_temp[pos...] = array(ttn[ll,pp])[pos...]
        end
        ttn_new[ll][pp] = (ll == number_of_layers(net)) ? ITensor(elT, data_temp, codom...) : ITensor(elT, data_temp, codom..., dom)
    end

    ortho_direction = _initialize_ortho_direction(net)
    ttnc = TreeTensorNetwork(ttn_new, ortho_direction, [-1,-1], net)

    if orthogonalize
        ttnc = _reorthogonalize!(ttnc, normalize = normalize)
    end

    return ttnc
end

function increase_dim_tree_tensor_network_randn(ttn::TreeTensorNetwork; maxdim::Int = 1,
    orthogonalize::Bool = true, normalize::Bool = orthogonalize, factor::Float64 = 10e-12, _elT = eltype(ttn))

    net = network(ttn)
    elT = promote_type(_elT, eltype(ttn))

    # if (!sectortype(net) == Trivial)
    #     throw(ArgumentError("Increasing subspace by filling with zeros only allowed without Quantum Numbers"))
    # end
    
    ttn_new = _initialize_empty_ttn(net)

    domains, codomains = _build_domains_and_codomains(net, maxdim)

    for (ll,pp) in NodeIterator(net)
        dom  = domains[ll][pp]
        codom = codomains[ll][pp]
        data_temp = (ll == number_of_layers(net)) ? factor.*randn(elT, ITensors.dim.((codom...,))) : factor.*randn(elT, ITensors.dim.((codom..., dom))) 

        pos_it = Iterators.product(UnitRange.(1, ITensors.dim.(inds(ttn[ll,pp])))...)
        for pos in pos_it
          data_temp[pos...] += array(ttn[ll,pp])[pos...]
        end
        ttn_new[ll][pp] = (ll == number_of_layers(net)) ? ITensor(elT, data_temp, codom...) : ITensor(elT, data_temp, codom..., dom)
    end

    ortho_direction = _initialize_ortho_direction(net)
    ttnc = TreeTensorNetwork(ttn_new, ortho_direction, [-1,-1], net)

    # if orthogonalize
    #     ttnc = _reorthogonalize!(ttnc, normalize = normalize)
    # end

    return ttnc
end
