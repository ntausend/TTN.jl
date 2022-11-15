

function _initialize_empty_ttn(net)
    ttn = Vector{Vector{TensorMap}}(undef, number_of_layers(net))
    foreach(eachlayer(net)) do ll
        ttn[ll] = Vector{TensorMap}(undef, number_of_tensors(net, ll))
    end
    return ttn
end


function _construct_product_tree_tensor_network(net::AbstractNetwork, states::Vector{<:AbstractString}, elT::DataType)

    ttn = _initialize_empty_ttn(net)   

    state_vecs = map(zip(physical_lattice(net), states)) do (nd,st)
        state(nd, st)
    end

    for pp in eachindex(net,1)
        chnds = child_nodes(net,(1,pp))
        state_vec = map(chnds) do (_,pc)
            state_vecs[pc]
        end
        state_comp = reduce(⊗, state_vec[2:end], init = state_vec[1])
        fused_dom = fuse(domain(state_comp))
        ttn[1][pp] = TensorMap(convert(Array, state_comp), codomain(state_comp) ← fused_dom)
    end
    elT = promote_type(eltype(ttn[1][1]), elT)

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



function _construct_random_tree_tensor_network(net::AbstractNetwork, maxdim::Int, elT::DataType) 
    ttn = _initialize_empty_ttn(net)
    domains, codomains = _build_domains_and_codomains(net,  maxdim)
    # we need to correct the last domain
    fused_last = fuse(codomains[end][1])
    # last domain should simply be one dimensional fixin the network
    domains[end][1] = _correct_domain(fused_last, 1)

    for (ll,pp) in NodeIterator(net)
        dom  = domains[ll][pp]
        codom = codomains[ll][pp]
        ttn[ll][pp] = TensorMap(randn, elT, codom ← dom)
    end
    return ttn
end

function _construct_random_tree_tensor_network(net::AbstractNetwork, target_charge, maxdim::Int, elT::DataType, tries::Int) 
    ttn = _initialize_empty_ttn(net)
    domains, codomains = _build_domains_and_codomains(net, target_charge, maxdim, tries)

    for (ll,pp) in NodeIterator(net)
        dom  = domains[ll][pp]
        codom = codomains[ll][pp]
        ttn[ll][pp] = TensorMap(randn, elT, codom ← dom)
    end
    return ttn
end



function _construct_random_tree_tensor_network2(net::AbstractNetwork{<:AbstractLattice{D,S, Trivial}},
                                                maxdim::Int, elT::DataType) where {D, S}
    # we dont need tensors for the physical layer... only for the virutal Tree Layers
    n_layers = number_of_layers(net)

    ttn = _initialize_empty_ttn(net)

    # getting the domain of the physical lattice
    phys_lat = physical_lattice(net)
    codomains = map(x -> hilbertspace(node(phys_lat,x)), eachindex(phys_lat))
    sp     = spacetype(node(phys_lat, 1))
    # iterate through the network to build the tensor tree
    for ll in eachlayer(net)
        domains = similar(codomains, number_of_tensors(net,ll))
        for pp in eachindex(net,ll)
            #d_n = dim(domain)^n_childNodes(net,(ll,pp))
            chd = child_nodes(net, (ll,pp))
            codomain_ch = map(chd) do (_, pc)
                codomains[pc]
            end
            d_n = prod(map(dim, codomain_ch))
            dom_dim = ll == n_layers ? 1 : min(d_n, maxdim)
            domains[pp] = sp(dom_dim)
            codomain_t = ProductSpace(codomain_ch...)
            ttn[ll][pp] = TensorMap(randn, elT, codomain_t ← domains[pp])
        end
        codomains = domains
    end
    return ttn
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
    dimsec = map(s -> dim(domain_new, s), sects)
    # now build a multinomial distribution wich draws maxdim samples
    ps = dimsec ./ dim_total
    rng = Multinomial(maxdim, ps)
    dims_new = rand(rng)
    new_irreps = map(zip(sects, dims_new)) do (s, d)
                    s => d
                end
    
    return spacetype(domain_new)(new_irreps)
end


function _build_domains_and_codomains(net::AbstractNetwork, maxdim::Int)
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

function _build_domains_and_codomains(net::AbstractNetwork{<:AbstractLattice{D,S,I}}, 
                                      target_charge::I, maxdim::Int, tries::Int) where{D,S, I<:Sector}
#function _build_domains_and_codomains(net::AbstractNetwork{D,S,I}, target_charge::I, maxdim::Int, tries::Int) where{D,S, I<:Sector}
    #if(I == Trivial)
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