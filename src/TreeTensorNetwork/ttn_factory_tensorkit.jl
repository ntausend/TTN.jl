function _initialize_empty_ttn(net::AbstractNetwork{L, TensorKitBackend}) where{L}
    ttn = Vector{Vector{TensorMap}}(undef, number_of_layers(net))
    foreach(eachlayer(net)) do ll
        ttn[ll] = Vector{TensorMap}(undef, number_of_tensors(net, ll))
    end
    return ttn
end

function _construct_product_tree_tensor_network(net::AbstractNetwork{L, TensorKitBackend}, states::Vector{<:AbstractString}, _elT::DataType) where{L}

    if length(states) != length(physical_lattice(net))
        throw(DimensionMismatch("Number of physical sites and and initial vals don't match"))
    end

    ttn = _initialize_empty_ttn(net)   

    state_vecs = map(zp -> state(zp[1], zp[2]), zip(physical_lattice(net), states))

    for pp in eachindex(net, 1)
        chnds = child_nodes(net, (1,pp))
        state_vec = map(n -> state_vecs[n[2]], chnds)
        state_comp = reduce(⊗, state_vec[2:end], init = state_vec[1])
        fused_dom = fuse(domain(state_comp))
        ttn[1][pp] = TensorMap(convert(Array, state_comp), codomain(state_comp) ← fused_dom)
    end
    elT = promote_type(eltype(ttn[1][1]), _elT)
    ttn[1][:] = one(elT) .* ttn[1][:]
    
    for ll in Iterators.drop(eachlayer(net),1)
        for pp in eachindex(net,ll)
            chnds = child_nodes(net,(ll,pp))
            codom_vec = map(chnds) do (_,pc)
                domain(ttn[ll-1][pc])
            end
            codom_comp = reduce(⊗, codom_vec[2:end], init = codom_vec[1])
            fused_dom  = fuse(codom_comp)
            ttn[ll][pp] = TensorMap(ones, elT, codom_comp ← fused_dom)
        end
    end

    return ttn
end


function _construct_random_tree_tensor_network(net::AbstractNetwork{L, TensorKitBackend}, maxdim::Int, elT::DataType) where{L}
    ttn = _initialize_empty_ttn(net)
    domains, codomains = _build_domains_and_codomains(net,  maxdim)
    # we need to correct the last domain
    fused_last = fuse(codomains[end][1])
    # last domain should simply be one dimensional fixing the network
    domains[end][1] = _correct_domain(fused_last, 1)

    for (ll,pp) in NodeIterator(net)
        dom  = domains[ll][pp]
        codom = codomains[ll][pp]
        ttn[ll][pp] = TensorMap(randn, elT, codom ← dom)
    end
    return ttn
end

function _build_domains_and_codomains(net::AbstractNetwork{L, TensorKitBackend}, maxdim::Int) where{L}
    phys_lat = physical_lattice(net)

    codomains = Vector{Vector{VectorSpace}}(undef, number_of_layers(net))
    domains = similar(codomains)
    foreach(eachlayer(net)) do ll
        codomains[ll] = Vector{VectorSpace}(undef, number_of_tensors(net, ll))
        domains[ll]   = similar(codomains[ll]) 
    end

    codomains_cur = map(x -> hilbertspace(node(phys_lat,x)), eachindex(phys_lat))
    for ll in eachlayer(net)
        for pp in eachindex(net, ll)
            chd = child_nodes(net, (ll,pp))
            # collect the codomains of the childs
            codomain_ch = map(chd) do (_, pc)
                codomains_cur[pc]
            end
            # fusing the codomains to get the new domain
            domain_new = fuse(codomain_ch...)
            # correct the domain to have maximal dimension of maxdim
            # and save the result
            domains[ll][pp] = _correct_domain(domain_new, maxdim)
            # form the product child space
            codomains[ll][pp] = ProductSpace(codomain_ch...)
            # save the new tensor
        end
        # save the current domains as the new codomains
        codomains_cur = domains[ll]
    end

    return domains, codomains
end

function _correct_domain(domain_new::VectorSpace, maxdim::Int)
    # getting current dimension of domain
    dim_total = dim(domain_new)

    # we are below the maxdim, so fast return
    dim_total ≤ maxdim && return domain_new

    sectortype(domain_new) == Trivial && return spacetype(domain_new)(maxdim)



    # get the different sectors
    sects = collect(sectors(domain_new))
    
    # get the dimensionality of the sectors
    dimsec = map(s -> dim_tk(domain_new, s), sects)
    # now build a multinomial distribution wich draws maxdim samples
    ps = dimsec ./ dim_total
    rng = Multinomial(maxdim, ps)
    dims_new = rand(rng)
    new_irreps = map(zip(sects, dims_new)) do (s, d)
                    s => d
                end
    # test this behavior, should be correct
    idx_zero = findall(iszero, dims_new)
    if !isnothing(idx_zero) 
        dims_new = deleteat!(dims_new, idx_zero)
        sects    = deleteat!(sects, idx_zero)
    end
    
    return spacetype(domain_new)(new_irreps)
end

function _construct_random_tree_tensor_network(net::AbstractNetwork{L, TensorKitBackend},
                                              target_charge, maxdim::Int, elT::DataType, tries::Int) where{L} 
    ttn = _initialize_empty_ttn(net)
    domains, codomains = _build_domains_and_codomains(net, target_charge, maxdim, tries)

    for (ll,pp) in NodeIterator(net)
        dom  = domains[ll][pp]
        codom = codomains[ll][pp]
        ttn[ll][pp] = TensorMap(randn, elT, codom ← dom)
    end
    return ttn
end


function _build_domains_and_codomains(net::AbstractNetwork{<:AbstractLattice{D,S,I, TensorKitBackend}}, 
                                      target_charge::I, maxdim::Int, tries::Int) where{D,S, I<:Sector}
    if(I == Trivial)
        target_sp = spacetype(net)(1)
    else
        target_sp = spacetype(net)(target_charge => 1)
    end

    codomains = Vector{Vector{VectorSpace}}(undef, number_of_layers(net))
    domains = similar(codomains)
    foreach(eachlayer(net)) do ll
        codomains[ll] = Vector{VectorSpace}(undef, number_of_tensors(net, ll))
        domains[ll]   = similar(codomains[ll]) 
    end

    for tt in 1:tries

        domains, codomains = _build_domains_and_codomains(net, maxdim)
        # correct the last domain
        domains[end][end] = target_sp

        I == Trivial &&  break
        #fusing the last codomains and see if target_space is contained
        fused_last = fuse(codomains[end][1])
        isempty(intersect(sectors(fused_last), sectors(target_sp))) ||  break

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


function increase_dim_tree_tensor_network_zeros(ttn::TreeTensorNetwork{L, TensorKitBackend}; maxdim::Int = 1,
    orthogonalize::Bool = true, normalize::Bool = orthogonalize, _elT = eltype(ttn)) where {L}

    net = network(ttn)
    elT = promote_type(_elT, eltype(ttn))
    
    if (!(sectortype(net) == Trivial))
        throw(ArgumentError("Increasing subspace by filling with zeros only allowed without Quantum Numbers"))
    end
    
    ttn_vec = _initialize_empty_ttn(net)

    domains, codomains = _build_domains_and_codomains(net,  maxdim)
    fused_last = fuse(codomains[end][1])
    domains[end][1] = _correct_domain(fused_last, 1) 

    
    for (ll,pp) in NodeIterator(net)

        prev_dom  = domain(ttn[(ll,pp)])
        prev_codom = codomain(ttn[(ll,pp)])

        dom  = domains[ll][pp]
        codom = codomains[ll][pp]

        data = zeros(elT, (dims(codom)..., dim(dom)))
        for i in 1:dim(prev_dom)
            data[1:dims(prev_codom)[1], 1:dims(prev_codom)[2], i] .= convert(Array, ttn[(ll,pp)])[:,:,i]
        end

        tensor = TensorMap(data, codom ← dom)

        ttn_vec[ll][pp] = tensor
    end

    ortho_direction = _initialize_ortho_direction(net)
    ttnc = TreeTensorNetwork(ttn_vec, ortho_direction, [-1,-1], net)

    if orthogonalize
        ttnc = _reorthogonalize!(ttnc, normalize = normalize)
    end

    return ttnc
end

function increase_dim_tree_tensor_network_randn(ttn::TreeTensorNetwork{L, TensorKitBackend}; maxdim::Int = 1,
                orthogonalize::Bool = true, normalize::Bool = orthogonalize, factor::Float64 = 10e-12, _elT = eltype(ttn)) where{L}

    net = network(ttn)
    elT = promote_type(_elT, eltype(ttn))
    
    ttn_vec = _initialize_empty_ttn(net)
    domains, codomains = _build_domains_and_codomains(net,  maxdim)
    fused_last = fuse(codomains[end][1])
    domains[end][1] = _correct_domain(fused_last, 1) 
    
    for (ll,pp) in NodeIterator(net)
        prev_dom  = domain(ttn[(ll,pp)])
        prev_codom = codomain(ttn[(ll,pp)])

        dom  = domains[ll][pp]
        codom = codomains[ll][pp]

        data = randn(elT, (dims(codom)..., dim(dom))).*factor
        for i in 1:dim(prev_dom)
            data[1:dims(prev_codom)[1], 1:dims(prev_codom)[2], i] .= convert(Array, ttn[(ll,pp)])[:,:,i]
        end

        tensor = TensorMap(data, codom ← dom)
        ttn_vec[ll][pp] = tensor
    end

    ortho_direction = _initialize_ortho_direction(net)
    ttnc = TreeTensorNetwork(ttn_vec, ortho_direction, [-1,-1], net)

    # if orthogonalize
    #     ttnc = _reorthogonalize!(ttnc, normalize = normalize)
    # end

    return ttnc
end