struct TreeTensorNetwork{D, S<:IndexSpace, I<:Sector}
    data::Vector{Vector{TensorMap}}
    ortho_center::Vector{Int}
    net::AbstractNetwork{D, S, I}
end

function eltype(ttn::TreeTensorNetwork) 
    elt_t = map(x -> eltype.(x), ttn.data)
    promote_type(vcat(elt_t...)...)
end

include("./ttn_factory.jl")

function RandomTreeTensorNetwork(net::AbstractNetwork; maxdim::Int = 1,
                orthogonalize::Bool = true, normalize::Bool = orthogonalize, elT = ComplexF64)
    ttn_vec = _construct_random_tree_tensor_network(net, maxdim, elT)
    ttn = TreeTensorNetwork(ttn_vec, [-1,-1], net)
    if orthogonalize
        ttn = _reorthogonalize!(ttn, normalize = normalize)
    end
    return ttn
end

function RandomTreeTensorNetwork(net::AbstractNetwork, target_charge; maxdim::Int = 1,
                orthogonalize::Bool = true, normalize::Bool = orthogonalize, elT = ComplexF64,
                tries::Int = 1000)
    ttn_vec = _construct_random_tree_tensor_network(net, target_charge, maxdim, elT, tries)
    ttn = TreeTensorNetwork(ttn_vec, [-1,-1], net)
    if orthogonalize
        ttn = _reorthogonalize!(ttn, normalize = normalize)
    end
    return ttn
end

function ProductTreeTensorNetwork(net::AbstractNetwork, states::Vector{<:AbstractString};
                orthogonalize::Bool = true, normalize::Bool = orthogonalize, elT = ComplexF64)
    ttn_vec = _construct_product_tree_tensor_network(net, states, elT)
    ttn = TreeTensorNetwork(ttn_vec, [-1,-1], net)
    if orthogonalize
        ttn = _reorthogonalize!(ttn, normalize = normalize)
    end
    return  ttn
end

function increase_dim_tree_tensor_network(ttn::TreeTensorNetwork; maxdim::Int = 1, elT = ComplexF64)

    net = network(ttn)
    
    ttnc = _initialize_empty_ttn(net)

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

        ttnc[ll][pp] = tensor
    end

    return TreeTensorNetwork(ttnc, [-1,-1], net)
end

# returning the ll-th tensor network layer
layer(ttn::TreeTensorNetwork, l::Int) = ttn.data[l]
number_of_layers(ttn::TreeTensorNetwork) = length(ttn.data)
# returning the network
network(ttn::TreeTensorNetwork) = ttn.net
# returning the current orthogonality center
ortho_center(ttn::TreeTensorNetwork) = Tuple(ttn.ortho_center)


Base.getindex(ttn::TreeTensorNetwork, l::Int, p::Int) = ttn.data[l][p]
Base.getindex(ttn::TreeTensorNetwork, pos::Tuple{Int, Int}) = getindex(ttn,pos[1],pos[2])

function Base.setindex!(ttn::TreeTensorNetwork, tn::TensorMap, l::Int, p::Int)
    ttn.data[l][p] = tn
    return ttn
end
Base.setindex!(ttn::TreeTensorNetwork, tn::TensorMap, pos::Tuple{Int, Int}) = setindex!(ttn, tn, pos[1], pos[2])


# makes `pos` orthogonal by splitting between domain and codomain as T = QR and shifting
# R into the parent node
_orthogonalize_to_parent!(ttn::TreeTensorNetwork, pos::Tuple{Int, Int}) = _orthogonalize_to_parent!(ttn, network(ttn), pos)

# general function for arbitrary Abstract Networks, maybe specified by special networks like binary trees etc
function _orthogonalize_to_parent!(ttn::TreeTensorNetwork, net::AbstractNetwork, pos::Tuple{Int, Int})
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

    idx_dom, idx_codom = split_index(net, pos, idx)
    
    perm = vcat(idx_dom..., idx_codom...)
    res = R*TensorKit.permute(tn_parent, idx_dom, idx_codom)
    res = TensorKit.permute(res, Tuple(perm[1:end-1]), (perm[end],))

    ttn[pos] = Q
    ttn[pos_parent] = res

    return ttn
end


# orhtogonalize towards the n-th child of this node
_orthogonalize_to_child!(ttn::TreeTensorNetwork, pos::Tuple{Int, Int}, n_child::Int) = _orthogonalize_to_child!(ttn, network(ttn), pos, n_child) 

# general function for arbitrary Abstract Networks, maybe specified by special networks like binary trees etc
function _orthogonalize_to_child!(ttn::TreeTensorNetwork, net::AbstractNetwork, pos::Tuple{Int, Int}, n_child::Int)
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
    return ttn
end


function _reorthogonalize!(ttn::TreeTensorNetwork; normalize::Bool = true)
    for pos in network(ttn)
        ttn = _orthogonalize_to_parent!(ttn, pos)
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

    _orthogonalize_to_parent!(ttn, oc)
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

function move_ortho!(ttn::TreeTensorNetwork, pos::Tuple{Int, Int}; normalize::Bool = false)
    check_valid_position(network(ttn), pos)

    # if ttn was not canonical, we simply reorthogonalize the ttn... is this ok?
    ortho_center(ttn) == (-1,-1) && _reorthogonalize!(ttn, normalize = normalize)
    
    oc = ortho_center(ttn)

    path = connecting_path(network(ttn), oc, pos)

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


# rework
function check_normality(ttn::TreeTensorNetwork)
    oc = ortho_center(ttn)
    all(oc .== -1) && return false, nothing
    
    lc = oc[1]
    pc = oc[2]
    
    net = network(ttn)

    n_layers = number_of_layers(net)
    
    
    are_id = Bool[]
    
    # general strategy for checking normalization
    for pos in net
        # for a general node check the nearest path connecting this node to
        # the orhtogonality centrum. The first node in this path dictates the 
        # orhtonomality flow, i.e. if the index set is divided between child nodes (1,..,nc)
        # and parent node (nc + 1), the tensor is written as A_{(i_1,...,i_nc), i_{nc+1}} 
        # because of the convention of our tensors where all child nodes are in the codomain
        # and the parent node is the domain. The orthonomality condition now dictates that contracting
        # the tensor over all indices not connecting towards the orthogonality center leads to an
        # identity over that index which is connecting towards the orhtogonality centrum.

        c_path = connecting_path(net, pos, oc)
        isempty(c_path) && continue # this is the orthognoality centrum, no checks here needed.
        nd_next = c_path[1]

        tn = ttn[pos]
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

        res = adjoint(tn)*tn
        push!(are_id, res ≈ one(res))
    end

    
    # finally calculate the norm on the orthogonality centrum
    res = TensorKit.norm(ttn[oc])
    return all(are_id), res
end

function Base.copy(ttn::TreeTensorNetwork)
    datac = deepcopy(ttn.data)
    ortho_centerc = deepcopy(ttn.ortho_center)
    netc = deepcopy(ttn.net)
    return TreeTensorNetwork(datac, ortho_centerc, netc)
end