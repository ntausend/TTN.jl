

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

function _construct_random_tree_tensor_network(net::AbstractNetwork{D,S, Trivial}, maxdim::Int, elT::DataType) where {D, S}
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
