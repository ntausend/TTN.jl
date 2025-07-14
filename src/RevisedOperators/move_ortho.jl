# makes `pos` orthogonal by splitting between domain and codomain as T = QR and shifting
# R into the parent node
_orthogonalize_to_parent!(ttn::TreeTensorNetwork, pos::Tuple{Int, Int}, node_cache::Dict; regularize = false) = _orthogonalize_to_parent!(ttn, network(ttn), pos, node_cache; regularize = regularize)

function _orthogonalize_to_parent!(ttn::TreeTensorNetwork, net::AbstractNetwork, pos::Tuple{Int, Int}, node_cache::Dict; regularize = false)
    @assert 0 < pos[1] ≤ number_of_layers(net)

    pos[1] == number_of_layers(net) && (return ttn)

    # getting the child tensor
    tn_child = haskey(node_cache, pos) ? node_cache[pos] : gpu(ttn[pos])
    # getting the parent node
    pos_parent = parent_node(net, pos)
    tn_parent = haskey(node_cache, pos_parent) ? node_cache[pos_parent] : gpu(ttn[pos_parent])

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
    ttn[pos] = cpu(Q)
    node_cache[pos] = Q

    ttn[pos_parent] = cpu(res)
    node_cache[pos_parent] = res

    ttn.ortho_direction[pos[1]][pos[2]] = number_of_child_nodes(net, pos) + 1
    ttn.ortho_direction[pos_parent[1]][pos_parent[2]] = -1

    return ttn
end

# orhtogonalize towards the n-th child of this node
_orthogonalize_to_child!(ttn::TreeTensorNetwork, pos::Tuple{Int, Int}, n_child::Int, node_cache::Dict) = _orthogonalize_to_child!(ttn, network(ttn), pos, n_child, node_cache) 

# general function for arbitrary Abstract Networks, maybe specified by special networks like binary trees etc
function _orthogonalize_to_child!(ttn::TreeTensorNetwork, net::AbstractNetwork, pos::Tuple{Int, Int}, n_child::Int, node_cache::Dict)
    # change to good Exception type, also proper handle of pos[1] being the lowest layer...
    @assert 0 < n_child ≤ number_of_child_nodes(net, pos)
    @assert 0 < pos[1] ≤ number_of_layers(net)

    pos[1] == 1 && (return ttn)
    
    # getting child position
    pos_child = child_nodes(net, pos)[n_child]
    
    # getting tensors
    tn_parent = haskey(node_cache, pos) ? node_cache[pos] : gpu(ttn[pos])
    tn_child = haskey(node_cache, pos_child) ? node_cache[pos_child] : gpu(ttn[pos_child])


    # the indices we need to have of the left hand from the splitting needs to be all except
    # the chared index to the child index
    idx_r = commonind(tn_parent, tn_child)
    idx_l = uniqueinds(tn_parent, idx_r)
    #Q,R = qr(tn_parent, idx_l, tags = tags(idx_r))
    Q,R  = factorize(tn_parent, idx_l; tags = tags(idx_r))

    res = tn_child * R
    ttn[pos_child] = cpu(res)
    node_cache[pos_child] = res

    ttn[pos] = cpu(Q)
    node_cache[pos] = Q

    ttn.ortho_direction[pos[1]][pos[2]] = n_child
    ttn.ortho_direction[pos_child[1]][pos_child[2]] = -1
    return ttn
end


function _reorthogonalize!(ttn::TreeTensorNetwork, node_cache::Dict; normalize::Bool = true)
    for pos in NodeIterator(network(ttn))
        ttn = _orthogonalize_to_parent!(ttn, pos, node_cache::Dict; regularize = normalize)
    end
    ttn.ortho_center .= [number_of_layers(ttn), 1]
    if(normalize)
        tn = ttn[ortho_center(ttn)]
        ttn[ortho_center(ttn)] = tn/norm(tn)
    end
    return ttn
end

function move_up!(ttn::TreeTensorNetwork, node_cache::Dict; normalize::Bool = false)
    
    # if ttn was not canonical, we simply reorthogonalize the ttn... is this ok?
    ortho_center(ttn) == (-1,-1) && _reorthogonalize!(ttn, normalize = normalize, node_cache)
    
    oc = ortho_center(ttn)
    pnd = parent_node(network(ttn), oc)
    if(!isnothing(pnd))
        ttn.ortho_center[1] = pnd[1]
        ttn.ortho_center[2] = pnd[2]
    end

    _orthogonalize_to_parent!(ttn, oc, node_cache; regularize = normalize)
end

function move_down!(ttn::TreeTensorNetwork, n_child::Int, node_cache::Dict; normalize::Bool = false)
    
    # if ttn was not canonical, we simply reorthogonalize the ttn... is this ok?
    ortho_center(ttn) == (-1,-1) && _reorthogonalize!(ttn, normalize = normalize, node_cache)
    
    oc = ortho_center(ttn)
    chnds = child_nodes(network(ttn), oc)
    if(!isnothing(chnds))
        child_node = chnds[n_child]
        ttn.ortho_center[1] = child_node[1]
        ttn.ortho_center[2] = child_node[2]
    end

    _orthogonalize_to_child!(ttn, oc, n_child, node_cache)
end

"""
```julia
    move_ortho!(ttn::TreeTensorNetwork, pos_target::Tuple{Int,Int}; normalize::Bool = false)
```

Shifts the orthogonality center to the `pos_target` position insite the network. If `normalize` is true, the state will be renormalized after moveing the orthogonality center.
If the tensor network is not orthogonalized at all, it will be orthogonalized first.
"""
function move_ortho!(ttn::TreeTensorNetwork, pos_target::Tuple{Int, Int}, node_cache::Dict; normalize::Bool = false)
    check_valid_position(network(ttn), pos_target)

    # if ttn was not canonical, we simply reorthogonalize the ttn... is this ok?
    ortho_center(ttn) == (-1,-1) && _reorthogonalize!(ttn, node_cache; normalize = normalize)
    
    oc = ortho_center(ttn)

    path = connecting_path(network(ttn), oc, pos_target)

    for pos in path
        Δoc = pos .- oc
        if(Δoc[1] == 1)
            # valid exception
            @assert pos == parent_node(network(ttn), oc)
            ttn = move_up!(ttn, node_cache)
        elseif(Δoc[1] == -1)
            n_child = index_of_child(network(ttn), pos)
            ttn = move_down!(ttn, n_child, node_cache)
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