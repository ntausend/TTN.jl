#struct TreeTensorNetwork{D, S<:IndexSpace, I<:Sector}
struct TreeTensorNetwork{N<:AbstractNetwork, T, B<:AbstractBackend}
    data::Vector{Vector{T}}
    # should we simply save the index of the tensor or
    # more information? Currently the index... not so happy about that
    ortho_direction::Vector{Vector{Int}}
    ortho_center::Vector{Int}
    net::N

    function TreeTensorNetwork(data::Vector{Vector{T}}, 
                               ortho_direction::Vector{Vector{Int}}, 
                               ortho_center::Vector{Int}, net::N) where{T, N} 
        new{N, T, backend(net)}(data, ortho_direction, ortho_center, net)
    end
end

function eltype(ttn::TreeTensorNetwork) 
    elt_t = map(x -> eltype.(x), ttn.data)
    promote_type(vcat(elt_t...)...)
end

backend(::Type{<:TreeTensorNetwork{N,T,B}}) where{N,T,B} = B
backend(ttn::TreeTensorNetwork) = backend(typeof(ttn))


function sites(ttn::TreeTensorNetwork{N,ITensor}) where N
    net = network(ttn)
    return map(eachindex(net, 0)) do pp
        prnt_nd = parent_node(net, (0,pp))
        only(inds(ttn[prnt_nd]; tags = "Site,n=$pp"))
    end
end

include("./ttn_factory_tensorkit.jl")
include("./ttn_factory_itensors.jl")

ITensors.maxlinkdim(ttn::TreeTensorNetwork) = maximum(map(pos -> maximum(ITensors.dims(ttn[pos])), NodeIterator(network(ttn))))

ITensors.siteinds(ttn::TreeTensorNetwork) = siteinds(physical_lattice(network(ttn)))


function _initialize_ortho_direction(net)
    ortho_direction = Vector{Vector{Int64}}(undef, number_of_layers(net))
    foreach(eachlayer(net)) do ll
        ortho_direction[ll] = repeat([-1], number_of_tensors(net, ll))
    end
    return ortho_direction
end

function RandomTreeTensorNetwork(net::AbstractNetwork; maxdim::Int = 1,
                orthogonalize::Bool = true, normalize::Bool = orthogonalize, elT = ComplexF64)
    ttn_vec = _construct_random_tree_tensor_network(net, maxdim, elT)
    ortho_direction = _initialize_ortho_direction(net)
    ttn = TreeTensorNetwork(ttn_vec, ortho_direction, [-1,-1], net)
    if orthogonalize
        ttn = _reorthogonalize!(ttn, normalize = normalize)
    end
    return ttn
end

function RandomTreeTensorNetwork(net::AbstractNetwork, target_charge; maxdim::Int = 1,
                orthogonalize::Bool = true, normalize::Bool = orthogonalize, elT = ComplexF64,
                tries::Int = 1000)
    ttn_vec = _construct_random_tree_tensor_network(net, target_charge, maxdim, elT, tries)
    ortho_direction = _initialize_ortho_direction(net)
    ttn = TreeTensorNetwork(ttn_vec, ortho_direction, [-1,-1], net)
    if orthogonalize
        ttn = _reorthogonalize!(ttn, normalize = normalize)
    end
    return ttn
end

function ProductTreeTensorNetwork(net::AbstractNetwork, states::Vector{<:AbstractString};
                orthogonalize::Bool = true, normalize::Bool = orthogonalize, elT = ComplexF64)
    ttn_vec = _construct_product_tree_tensor_network(net, states, elT)
    ortho_direction = _initialize_ortho_direction(net)
    ttn = TreeTensorNetwork(ttn_vec, ortho_direction, [-1,-1], net)
    if orthogonalize
        ttn = _reorthogonalize!(ttn, normalize = normalize)
    end
    return  ttn
end

function increase_dim_tree_tensor_network_zeros(ttn::TreeTensorNetwork; maxdim::Int = 1,
                orthogonalize::Bool = true, normalize::Bool = orthogonalize, elT = ComplexF64)

    net = network(ttn)
    
    ttn_vec = _initialize_empty_ttn(net)

    domains, codomains = _build_domains_and_codomains(net,  maxdim)
    fused_last = fuse(codomains[end][1])
    domains[end][1] = _correct_domain(fused_last, 1) 
    
    for (ll,pp) in net

        prev_dom  = domain(ttn[(ll,pp)])
        prev_codom = codomain(ttn[(ll,pp)])

        dom  = domains[ll][pp]
        codom = codomains[ll][pp]

        data = zeros((dims(codom)..., dim(dom)))
        for i in 1:dim(prev_dom)
            data[1:dims(prev_codom)[1], 1:dims(prev_codom)[2], i] = convert(Array, ttn[(ll,pp)])[:,:,i]
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

function increase_dim_tree_tensor_network_randn(ttn::TreeTensorNetwork; maxdim::Int = 1,
                orthogonalize::Bool = true, normalize::Bool = orthogonalize, factor::Float64 = 10e-12, elT = ComplexF64)

    net = network(ttn)
    
    ttn_vec = _initialize_empty_ttn(net)

    domains, codomains = _build_domains_and_codomains(net,  maxdim)
    fused_last = fuse(codomains[end][1])
    domains[end][1] = _correct_domain(fused_last, 1) 
    
    for (ll,pp) in net

        prev_dom  = domain(ttn[(ll,pp)])
        prev_codom = codomain(ttn[(ll,pp)])

        dom  = domains[ll][pp]
        codom = codomains[ll][pp]

        data = randn(elT, (dims(codom)..., dim(dom))).*factor
        for i in 1:dim(prev_dom)
            data[1:dims(prev_codom)[1], 1:dims(prev_codom)[2], i] = convert(Array, ttn[(ll,pp)])[:,:,i]
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

# returning the ll-th tensor network layer
layer(ttn::TreeTensorNetwork, l::Int) = ttn.data[l]
number_of_layers(ttn::TreeTensorNetwork) = length(ttn.data)
# returning the network
network(ttn::TreeTensorNetwork) = ttn.net
# returning the current orthogonality center
ortho_center(ttn::TreeTensorNetwork) = Tuple(ttn.ortho_center)
is_orthogonalized(ttn::TreeTensorNetwork) = all(ortho_center(ttn) .!= (-1,-1))

# returns the ortho_direction of a given position in the network.
# this returns the INDEX inside the tensor at that position which is connected to the
# ortho center, see the check_normality function for usage
ortho_direction(ttn::TreeTensorNetwork, pos::Tuple{Int, Int}) = ttn.ortho_direction[pos[1]][pos[2]]


Base.getindex(ttn::TreeTensorNetwork, l::Int, p::Int) = ttn.data[l][p]
Base.getindex(ttn::TreeTensorNetwork, pos::Tuple{Int, Int}) = getindex(ttn,pos[1],pos[2])

function Base.setindex!(ttn::TreeTensorNetwork{N,T}, tn::T, l::Int, p::Int) where{N,T}
    ttn.data[l][p] = tn
    return ttn
end
Base.setindex!(ttn::TreeTensorNetwork{N, T}, tn::T, pos::Tuple{Int, Int}) where{N,T} = setindex!(ttn, tn, pos[1], pos[2])

# makes `pos` orthogonal by splitting between domain and codomain as T = QR and shifting
# R into the parent node
_orthogonalize_to_parent!(ttn::TreeTensorNetwork, pos::Tuple{Int, Int}; regularize = false) = _orthogonalize_to_parent!(ttn, network(ttn), pos; regularize = regularize)

# general function for arbitrary Abstract Networks, maybe specified by special networks like binary trees etc
function _orthogonalize_to_parent!(ttn::TreeTensorNetwork{LA,TensorMap,TensorKitBackend}, net::AbstractNetwork, pos::Tuple{Int, Int}; regularize = false) where{LA}
    @assert 0 < pos[1] ≤ number_of_layers(net)

    pos[1] == number_of_layers(net) && (return ttn)

    # getting the position of the child in the parent nodes codomain
    idx = index_of_child(net, pos)

    # getting the child tensor
    tn_child = ttn[pos]
    # getting the parent node
    pos_parent = parent_node(net, pos)
    tn_parent = ttn[pos_parent]

    # now split the child tensor. Luckly it is already in the correct domain/codomain splitting
    # such that Q,R = leftorth(tn_child) has U as the new unitary for pos and R has to contracted to
    # corresponding leg of the parent node
    
    Q,R = leftorth(tn_child)
    
    # handles large normed TTN's. Specially for random initialization
    if regularize
        R = R/norm(R)
    end

    idx_dom, idx_codom = split_index(net, pos, idx)
    
    perm = vcat(idx_dom..., idx_codom...)
    res = R*TensorKit.permute(tn_parent, idx_dom, idx_codom)
    res = TensorKit.permute(res, Tuple(perm[1:end-1]), (perm[end],))

    ttn[pos] = Q
    ttn[pos_parent] = res

    ttn.ortho_direction[pos[1]][pos[2]] = number_of_child_nodes(net, pos) + 1
    ttn.ortho_direction[pos_parent[1]][pos_parent[2]] = -1

    return ttn
end

function _orthogonalize_to_parent!(ttn::TreeTensorNetwork{LA,ITensor,ITensorsBackend}, net::AbstractNetwork, pos::Tuple{Int, Int}; regularize = false) where{LA}
    @assert 0 < pos[1] ≤ number_of_layers(net)

    pos[1] == number_of_layers(net) && (return ttn)

    # getting the child tensor
    tn_child = ttn[pos]
    # getting the parent node
    pos_parent = parent_node(net, pos)
    tn_parent = ttn[pos_parent]

    # the left index for the splitting is simply the not commoninds of tn_child
    idx_r = commonind(tn_child, tn_parent)
    #idx_l = uniqueinds(tn_child, tn_parent)
    idx_l = uniqueinds(tn_child, idx_r)
    #Q,R = qr(tn_child, idx_l; tags = tags(idx_r))
    Q,R = factorize(tn_child, idx_l; tags = tags(idx_r))
    
    # handles large normed TTN's. Specially for random initialization
    if regularize
        R .= R./norm(R)
    end
    res = R*tn_parent
    ttn[pos] = Q
    ttn[pos_parent] = res

    ttn.ortho_direction[pos[1]][pos[2]] = number_of_child_nodes(net, pos) + 1
    ttn.ortho_direction[pos_parent[1]][pos_parent[2]] = -1

    return ttn
end

# orhtogonalize towards the n-th child of this node
_orthogonalize_to_child!(ttn::TreeTensorNetwork, pos::Tuple{Int, Int}, n_child::Int) = _orthogonalize_to_child!(ttn, network(ttn), pos, n_child) 

# general function for arbitrary Abstract Networks, maybe specified by special networks like binary trees etc
function _orthogonalize_to_child!(ttn::TreeTensorNetwork{LA,TensorMap,TensorKitBackend}, net::AbstractNetwork, pos::Tuple{Int, Int}, n_child::Int) where{LA}
    # change to good Exception type, also proper handle of pos[1] being the lowest layer...
    @assert 0 < n_child ≤ number_of_child_nodes(net, pos)
    @assert 0 < pos[1] ≤ number_of_layers(net)

    pos[1] == 1 && (return ttn)
    
    # getting child position
    pos_child = child_nodes(net, pos)[n_child]
    
    # getting tensors
    tn_parent = ttn[pos] 
    tn_child  = ttn[pos_child]

    # now we need to permute the inds such that the childs index ist the left most. Since then we can use
    # rightort yielding LQ decomposition where we can push L to the childes node afterwards
    idx_dom, idx_codom = split_index(net, pos, n_child)


    perm = vcat(idx_dom..., idx_codom...)

    L,Q = rightorth(tn_parent, idx_dom, idx_codom)

    ttn[pos_child] = tn_child * L

    ttn[pos] = TensorKit.permute(Q, Tuple(perm[1:end-1]), (perm[end],))

    ttn.ortho_direction[pos[1]][pos[2]] = n_child
    ttn.ortho_direction[pos_child[1]][pos_child[2]] = -1
    return ttn
end
# general function for arbitrary Abstract Networks, maybe specified by special networks like binary trees etc
function _orthogonalize_to_child!(ttn::TreeTensorNetwork{LA,ITensor,ITensorsBackend}, net::AbstractNetwork, pos::Tuple{Int, Int}, n_child::Int) where{LA}
    # change to good Exception type, also proper handle of pos[1] being the lowest layer...
    @assert 0 < n_child ≤ number_of_child_nodes(net, pos)
    @assert 0 < pos[1] ≤ number_of_layers(net)

    pos[1] == 1 && (return ttn)
    
    # getting child position
    pos_child = child_nodes(net, pos)[n_child]
    
    # getting tensors
    tn_parent = ttn[pos] 
    tn_child  = ttn[pos_child]

    # the indices we need to have of the left hand from the splitting needs to be all except
    # the chared index to the child index
    idx_r = commonind(tn_parent, tn_child)
    idx_l = uniqueinds(tn_parent, idx_r)
    #Q,R = qr(tn_parent, idx_l, tags = tags(idx_r))
    Q,R  = factorize(tn_parent, idx_l; tags = tags(idx_r))

    ttn[pos_child] = tn_child * R

    ttn[pos] = Q

    ttn.ortho_direction[pos[1]][pos[2]] = n_child
    ttn.ortho_direction[pos_child[1]][pos_child[2]] = -1
    return ttn
end


function _reorthogonalize!(ttn::TreeTensorNetwork; normalize::Bool = true)
    for pos in NodeIterator(network(ttn))
        ttn = _orthogonalize_to_parent!(ttn, pos; regularize = normalize)
    end
    ttn.ortho_center .= [number_of_layers(ttn), 1]
    if(normalize)
        tn = ttn[ortho_center(ttn)]
        ttn[ortho_center(ttn)] = tn/TensorKit.norm(tn)
    end
    return ttn
end

function move_up!(ttn::TreeTensorNetwork; normalize::Bool = false)
    
    # if ttn was not canonical, we simply reorthogonalize the ttn... is this ok?
    ortho_center(ttn) == (-1,-1) && _reorthogonalize!(ttn, normalize = normalize)
    
    oc = ortho_center(ttn)
    pnd = parent_node(network(ttn), oc)
    if(!isnothing(pnd))
        ttn.ortho_center[1] = pnd[1]
        ttn.ortho_center[2] = pnd[2]
    end

    _orthogonalize_to_parent!(ttn, oc; regularize = normalize)
end

function move_down!(ttn::TreeTensorNetwork, n_child::Int; normalize::Bool = false)
    
    # if ttn was not canonical, we simply reorthogonalize the ttn... is this ok?
    ortho_center(ttn) == (-1,-1) && _reorthogonalize!(ttn, normalize = normalize)
    
    oc = ortho_center(ttn)
    chnds = child_nodes(network(ttn), oc)
    if(!isnothing(chnds))
        child_node = chnds[n_child]
        ttn.ortho_center[1] = child_node[1]
        ttn.ortho_center[2] = child_node[2]
    end

    _orthogonalize_to_child!(ttn, oc, n_child)
end

function move_ortho!(ttn::TreeTensorNetwork, pos_target::Tuple{Int, Int}; normalize::Bool = false)
    check_valid_position(network(ttn), pos_target)

    # if ttn was not canonical, we simply reorthogonalize the ttn... is this ok?
    ortho_center(ttn) == (-1,-1) && _reorthogonalize!(ttn, normalize = normalize)
    
    oc = ortho_center(ttn)

    path = connecting_path(network(ttn), oc, pos_target)

    for pos in path
        Δoc = pos .- oc
        if(Δoc[1] == 1)
            # valid exception
            @assert pos == parent_node(network(ttn), oc)
            ttn = move_up!(ttn)
        elseif(Δoc[1] == -1)
            n_child = index_of_child(network(ttn), pos)
            ttn = move_down!(ttn, n_child)
        else
            # valid exception
            error("Invalid path connecting old orthogonality center with new one.. ", path)
        end
        oc = pos
    end
    ttn.ortho_center[1] = oc[1]
    ttn.ortho_center[2] = oc[2]
    return ttn
end

include("./utils_itensors.jl")
function adjust_tree_tensor_dimensions!(ttn::TreeTensorNetwork{N,ITensor}, maxdim::Int;
                                        use_random::Bool=true, reorthogonalize::Bool = true) where{N}
    # got through all the layers and adjusting the bond dimension of the tensors by inserting new larger tensors (if necessary)
    # filled with elements defined by `padding_function`

    net = network(ttn)
    # save the orhtogonality centrum for later reference
    oc = ortho_center(ttn)
    
    _reorthogonalize = oc != (-1, -1)
    reorthogonalize = _reorthogonalize && reorthogonalize

    for pos in NodeIterator(net)
        # building the fused leg defined by the child legs
        
        prnt_node = parent_node(net, pos)
        # top node, nothing to do here
        isnothing(prnt_node) && continue

        T = ttn[pos]

        # parent index, to be replaced, if necessary
        prnt_idx = commonind(T, ttn[prnt_node])
        # child indices, defining the replacement
        chld_idx = uniqueinds(T, prnt_idx)
        
        # check if dimension of parent link already is good

        dim_chlds = ITensors.dim.(chld_idx)
        dim_full = prod(dim_chlds)
        dim_prnt = ITensors.dim(prnt_idx)
       
        dim_prnt == dim_full || dim_prnt ≥ maxdim && continue
        #dim_prnt == min(dim_full, maxdim) && continue
        
        # permute the indeces to have the enlarging link at the last position
        T = ITensors.permute(T, chld_idx..., prnt_idx)
        leftover_idx = uniqueinds(ttn[prnt_node], prnt_idx)
        Tprnt = ITensors.permute(ttn[prnt_node], leftover_idx..., prnt_idx)
        

        if hasqns(T)
            qn_link = Index([flux(T)=>1], "QNL"; dir = ITensors.In)
        else
            qn_link = Index(1, "QNL")
        end

        # build the full link with possible flow of the qn numbers involved
        # build it in the opposite direction of the parent node to have
        # it formally aligned in terms of flows 
        full_link = combinedind(combiner(chld_idx..., qn_link; dir = dir(dag(prnt_idx))))
        # getting the complement of the current parent link in the full link
        link_complement = complement(dag(full_link), prnt_idx)
        # if dimension of complement link is zero, we simply throw it away
        iszero(dim(link_complement)) && continue

        # now correct the space with the missing number of dimensions
        pssbl_dim = min(maxdim, dim_full) # maximal possible link dimension allowed
        # correct sectors appearing in link_complement but not appearing in the current link to at least one sector
        link_padding = _correct_domain(link_complement, pssbl_dim-dim_prnt)
        if hasqns(T)
            # how to do this more smart? now we increasing the qn larger than necessary...
            link_missing = complement(link_complement, link_padding)
            prnt_secs = first.(space(prnt_idx))
            cmpl_sp = space(link_missing)
            sp_n_or_missing = map(cmpl_sp) do (qn, dd)
                qn ∉ prnt_secs && dd > 0 ? qn => 1 : missing
            end
            sp_n = vcat(collect(skipmissing(sp_n_or_missing)), space(link_padding))
            link_padding = Index(sp_n; tags = tags(link_padding), dir = dir(link_padding))
        end
        # link_new has now the same direction as prnt_idx, and therefore how it is attached to T
        link_new     = combinedind(combiner(directsum(prnt_idx, link_padding); tags = tags(prnt_idx)))

        Tn     = _enlarge_tensor(T, chld_idx, prnt_idx, link_new, use_random)
        # it has to be daggered for the parent_node to have it in the codomain
        #Tnprnt = _enlarge_tensor(Tprnt, leftover_idx, prnt_idx, dag(link_new), use_random)
        Tnprnt = _enlarge_tensor(Tprnt, leftover_idx, prnt_idx, dag(link_new), false)

        ttn[pos] = Tn
        ttn[prnt_node] = Tnprnt

        ttn.ortho_center .= [-1, -1]
    end
    # if ttn was orthogonal, restore the original oc center
    if reorthogonalize
        @show reorthogonalize
        ttn = _reorthogonalize!(ttn)
        ttn = move_ortho!(ttn, oc)
    end
    return ttn
end
adjust_tree_tensor_dimensions(ttn::TreeTensorNetwork, maxdim::Int; kwargs...) = adjust_tree_tensor_dimensions!(copy(ttn), maxdim; kwargs...)



# rework
function check_normality(ttn::TreeTensorNetwork{L, TensorMap, TensorKitBackend}) where{L}
    oc = ortho_center(ttn)
    all(oc .== -1) && return false, nothing
    
    net = network(ttn)

    
    are_id = Bool[]
    
    # general strategy for checking normalization
    for pos in NodeIterator(net)
        # for a general node check the nearest path connecting this node to
        # the orhtogonality centrum. The first node in this path dictates the 
        # orhtonomality flow, i.e. if the index set is divided between child nodes (1,..,nc)
        # and parent node (nc + 1), the tensor is written as A_{(i_1,...,i_nc), i_{nc+1}} 
        # because of the convention of our tensors where all child nodes are in the codomain
        # and the parent node is the domain. The orthonomality condition now dictates that contracting
        # the tensor over all indices not connecting towards the orthogonality center leads to an
        # identity over that index which is connecting towards the orhtogonality centrum.

        pos == ortho_center(ttn) && continue
        tn = ttn[pos]
        
        #=
        c_path = connecting_path(net, pos, oc)
        isempty(c_path) && continue # this is the orthognoality centrum, no checks here needed.
        nd_next = c_path[1]

        # in case of orhtogonality direction is towards the parent node -> the default grouping
        # is already the desired one for contracting
        if(nd_next[1] == pos[1] - 1)
            # in case of the orhtogonality direction is towards the lower layer, 
            # we first need the index number of the child
            idx_ch = index_of_child(net, nd_next)
            # and then construct the possible remaing indices
            idx_dom, idx_codom = split_index(net, pos, idx_ch)

            # now we can perform the desired permutation on the tensor
            tn = TensorKit.permute(tn, idx_codom, idx_dom)
        end
        =#
        ortho_dir = ortho_direction(ttn, pos)
        idx_dom, idx_codom = split_index(net, pos, ortho_dir)
        tn = TensorKit.permute(tn, idx_codom, idx_dom)

        res = adjoint(tn)*tn
        push!(are_id, res ≈ one(res))
    end

    
    # finally calculate the norm on the orthogonality centrum
    res = TensorKit.norm(ttn[oc])
    return all(are_id), res
end
function check_normality(ttn::TreeTensorNetwork{L, ITensor, ITensorsBackend}) where{L}
    oc = ortho_center(ttn)
    all(oc .== -1) && return false, nothing
    
    net = network(ttn)

    
    are_id = Bool[]
    
    # general strategy for checking normalization
    for pos in NodeIterator(net)
        # for a general node check the nearest path connecting this node to
        # the orhtogonality centrum. The first node in this path dictates the 
        # orhtonomality flow, i.e. if the index set is divided between child nodes (1,..,nc)
        # and parent node (nc + 1), the tensor is written as A_{(i_1,...,i_nc), i_{nc+1}} 
        # because of the convention of our tensors where all child nodes are in the codomain
        # and the parent node is the domain. The orthonomality condition now dictates that contracting
        # the tensor over all indices not connecting towards the orthogonality center leads to an
        # identity over that index which is connecting towards the orhtogonality centrum.

        pos == ortho_center(ttn) && continue
        tn = ttn[pos]
        
        #=
        c_path = connecting_path(net, pos, oc)
        isempty(c_path) && continue # this is the orthognoality centrum, no checks here needed.
        nd_next = c_path[1]

        # in case of orhtogonality direction is towards the parent node -> the default grouping
        # is already the desired one for contracting
        if(nd_next[1] == pos[1] - 1)
            # in case of the orhtogonality direction is towards the lower layer, 
            # we first need the index number of the child
            idx_ch = index_of_child(net, nd_next)
            # and then construct the possible remaing indices
            idx_dom, idx_codom = split_index(net, pos, idx_ch)

            # now we can perform the desired permutation on the tensor
            tn = TensorKit.permute(tn, idx_codom, idx_dom)
        end
        =#
        ortho_dir = ortho_direction(ttn, pos)
        if length(inds(tn)) == ortho_dir
            idx = commonind(tn, ttn[parent_node(net, pos)])
        else
            chd = child_nodes(net, pos)
            idx_chd = chd[index_of_child(net, (pos[1]-1, ortho_dir))]
            idx = commonind(tn, ttn[idx_chd])
        end
        
        res = tn * dag(prime(tn, idx))

        #idx_dom, idx_codom = split_index(net, pos, ortho_dir)
        #tn = TensorKit.permute(tn, idx_codom, idx_dom)

        #res = adjoint(tn)*tn
        push!(are_id, res ≈ delta(eltype(res), idx, prime(idx)))
    end

    
    # finally calculate the norm on the orthogonality centrum
    res = ITensors.norm(ttn[oc])
    return all(are_id), res
end

function Base.copy(ttn::TreeTensorNetwork)
    datac = deepcopy(ttn.data)
    ortho_centerc = deepcopy(ttn.ortho_center)
    netc = deepcopy(ttn.net)
    ortho_directionc = deepcopy(ttn.ortho_direction)
    return TreeTensorNetwork(datac, ortho_directionc, ortho_centerc, netc)
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ttn::TTNKit.TreeTensorNetwork)
    ortho_center = ttn.ortho_center
    ortho_direction = ttn.ortho_direction
    net = ttn.net
    data = ttn.data
        
    g = create_group(parent, name)

    ### save ortho_center ###
    write(g, "ortho_center", ortho_center)

    ### save ortho_direction ###
    group_ortho_center = create_group(g, "ortho_direction")
    for (layer, od_layer) in enumerate(ortho_direction)
        name_layer = "layer_"*string(layer)
        write(group_ortho_center, name_layer, od_layer)
    end

    ### save net ###
    write(g, "net", net)
        
    ### save data ###
    group_data = create_group(g, "data")
    for (layer, data_layer) in enumerate(data)
        name_data_layer = "layer_"*string(layer)
        group_data_layer = create_group(group_data, name_data_layer)
        for (node, data_node) in enumerate(data_layer)
            name_data_node = "node_"*string(node)
            write(group_data_layer, name_data_node, cpu(data_node))
        end
    end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{TTNKit.TreeTensorNetwork})
    g = open_group(parent, name)

    ### read net ###
    net = read(g, "net", TTNKit.AbstractNetwork)

    ### read data ###
    group_data = open_group(g, "data")
    data = map(1:number_of_layers(net)) do ll
        name_layer = "layer_$(ll)"
        group_data_layer = open_group(group_data, name_layer)
        map(1:number_of_tensors(net,ll)) do pp
            name_node = "node_$(pp)"
            read(group_data_layer, name_node, ITensor)
        end
    end

    ### read ortho_center ###
    ortho_center = read(g, "ortho_center")

    ### read ortho_direction ###
    group_od = open_group(g, "ortho_direction")
    ortho_direction = map(keys(group_od)) do layer
        read(group_od, layer)
    end

    return TTNKit.TreeTensorNetwork(data, ortho_direction, ortho_center, net)
end
