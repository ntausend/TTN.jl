struct TreeTensorNetwork
    data::Vector{Vector{TensorMap}}
    ortho_center::Vector{Int}
    net::AbstractNetwork
end


function construct_random_tree_tensor_network(net::AbstractNetwork)
    # we dont need tensors for the physical layer... only for the virutal Tree Layers
    number_of_layers = TTNKit.n_layers(net)

    ttn = Vector{Vector{TensorMap}}(undef, number_of_layers)
    foreach(1:number_of_layers) do ll
        ttn[ll] = Vector{TensorMap}(undef, TTNKit.n_tensors(net, ll))
    end

    # iterate through the network to build the tensor tree
    for pp in net
        domain = TTNKit.hilbertspace(net, pp)

        n_ch = childNodes(net, pp)
        sp   = map(n_ch) do(ch)
            TTNKit.hilbertspace(net, ch)
        end

        codomain = ProductSpace(sp...)

        ttn[pp[1]][pp[2]] = TensorMap(randn, codomain ← domain)
    end

    return ttn
end

function TreeTensorNetwork(net::AbstractNetwork; orthogonalize = true, normalize = orthogonalize)
    ttn_vec = construct_random_tree_tensor_network(net)
    ttn = TreeTensorNetwork(ttn_vec, [-1,-1], net)
    if orthogonalize
        ttn = _reorthogonalize!(ttn, normalize = normalize)
    end
    return  ttn
end

# returning the ll-th tensor network layer
layer(ttn::TreeTensorNetwork, l::Int) = ttn.data[l]
n_layers(ttn::TreeTensorNetwork) = length(ttn.data)
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

    pos[1] == TTNKit.n_layers(net) && (return ttn)

    # getting the position of the child in the parent nodes codomain
    idx = TTNKit.index_of_child(net, pos)

    # getting the child tensor
    tn_child = ttn[pos]
    # getting the parent node
    pos_parent = parentNode(net, pos)
    tn_parent = ttn[pos_parent]

    # now split the child tensor. Luckly it is already in the correct domain/codomain splitting
    # such that Q,R = leftorth(tn_child) has U as the new unitary for pos and R has to contracted to
    # corresponding leg of the parent node
    
    Q,R = leftorth(tn_child)

    allinds = 1:(1 + length(codomain(tn_parent)))
    res_inds = deleteat!(collect(allinds), idx)
    perm = vcat(idx, res_inds)
    res = R*TensorKit.permute(tn_parent, (idx,), Tuple(res_inds))
    res = TensorKit.permute(res, Tuple(perm[1:end-1]), (perm[end],))

    ttn[pos] = Q
    ttn[pos_parent] = res

    return ttn
end


# orhtogonalize towards the n-th child of this node
_orthogonalize_to_child!(ttn::TreeTensorNetwork, pos::Tuple{Int, Int}, n_child::Int) = _orthogonalize_to_child!(ttn, network(ttn), pos, n_child) 

# general function for arbitrary Abstract Networks, maybe specified by special networks like binary trees etc
function _orthogonalize_to_child!(ttn::TreeTensorNetwork, net::AbstractNetwork, pos::Tuple{Int, Int}, n_child::Int)
    @assert 0 < n_child ≤ n_childNodes(net, pos)

    pos[1] == 1 && (return ttn)
    
    # getting child position
    pos_child = TTNKit.childNodes(net, pos)[n_child]
    
    # getting tensors
    tn_parent = ttn[pos] 
    tn_child  = ttn[pos_child]

    # now we need to permute the inds such that the childs index ist the left most. Since then we can use
    # rightort yielding LQ decomposition where we can push L to the childes node afterwards
    allinds  = 1:(1 + length(codomain(tn_parent)))
    res_inds = deleteat!(collect(allinds), n_child)
    perm = vcat(n_child, res_inds)

    L,Q = rightorth(tn_parent, (n_child,), Tuple(res_inds))

    ttn[pos_child] = tn_child * L

    ttn[pos] = TensorKit.permute(Q, Tuple(perm[1:end-1]), (perm[end],))
    return ttn
end


function _reorthogonalize!(ttn::TreeTensorNetwork; normalize = true)
    for pos in network(ttn)
        ttn = _orthogonalize_to_parent!(ttn, pos)
    end
    ttn.ortho_center .= [n_layers(ttn), 1]
    if(normalize)
        tn = ttn[ortho_center(ttn)]
        ttn[ortho_center(ttn)] = tn/norm(tn)
    end
    return ttn
end


function move_up!(ttn::TreeTensorNetwork; normalize = false)
    
    # if ttn was not canonical, we simply reorthogonalize the ttn... is this ok?
    ortho_center(ttn) == (-1,-1) && _reorthogonalize!(ttn, normalize = normalize)
    
    oc = ortho_center(ttn)
    parent_node = TTNKit.parentNode(network(ttn), oc)
    if(!isnothing(parent_node))
        ttn.ortho_center[1] = parent_node[1]
        ttn.ortho_center[2] = parent_node[2]
    end

    _orthogonalize_to_parent!(ttn, oc)
end

function move_down!(ttn::TreeTensorNetwork, n_child::Int; normalize = false)
    
    # if ttn was not canonical, we simply reorthogonalize the ttn... is this ok?
    ortho_center(ttn) == (-1,-1) && _reorthogonalize!(ttn, normalize = normalize)
    
    oc = ortho_center(ttn)
    child_nodes = TTNKit.childNodes(network(ttn), oc)
    if(!isnothing(child_nodes))
        child_node = child_nodes[n_child]
        ttn.ortho_center[1] = child_node[1]
        ttn.ortho_center[2] = child_node[2]
    end

    _orthogonalize_to_child!(ttn, oc, n_child)
end

function move_ortho!(ttn::TreeTensorNetwork, pos::Tuple{Int, Int})
    @assert TTNKit.check_valid_pos(network(ttn), pos)

    # if ttn was not canonical, we simply reorthogonalize the ttn... is this ok?
    ortho_center(ttn) == (-1,-1) && _reorthogonalize!(ttn, normalize = normalize)
    
    oc = ortho_center(ttn)

    path = TTNKit.connectingPath(network(ttn), oc, pos)

    for pos in path
        Δoc = pos .- oc
        if(Δoc[1] == 1)
            @assert pos == TTNKit.parentNode(network(ttn), oc)
            ttn = move_up!(ttn)
        elseif(Δoc[1] == -1)
            n_child = TTNKit.index_of_child(network(ttn), pos)
            ttn = move_down!(ttn, n_child)
        else
            error("Invalid path connecting old orthogonality center with new one.. ", path)
        end
        oc = pos
    end
    ttn.ortho_center[1] = oc[1]
    ttn.ortho_center[2] = oc[2]
    return ttn
end



function check_normality(ttn::TreeTensorNetwork)
    oc = ortho_center(ttn)
    all(oc .== -1) && return false, nothing
    
    lc = oc[1]
    pc = oc[2]
    
    net = network(ttn)

    !(net isa BinaryChainNetwork) && 
        (error("Normality check only implemented for One Dimensional Binary Trees currently..."))

    number_of_layers = TTNKit.n_layers(net)
    
    
    are_id = Bool[]
    
    # check identities in layers below the orthogonal center
    for ll in 1:lc-1
        number_of_tensors = TTNKit.n_tensors(net, ll)
        for pp in 1:number_of_tensors
            tn = ttn[ll,pp]
            #@tensor res[-1; -2] := conj(tn[1, 2; -1]) * tn[1, 2; -2]
            res = adjoint(tn)*tn
            push!(are_id,res ≈ isomorphism(domain(res), codomain(res)))
        end
    end
    
    # check identities inside the layer of the orthogonal center
    number_of_tensors = TTNKit.n_tensors(net,lc)
    for pp in vcat(collect(1:pc-1), collect(pc+1:number_of_tensors))
        tn = ttn[lc,pp]
        #@tensor res[-1; -2] := conj(tn[1, 2; -1]) * tn[1, 2; -2]
        res = adjoint(tn)*tn
        push!(are_id,res ≈ isomorphism(domain(res), codomain(res)))
    end
    
    # check identities for tensors above the orthogonality center, luckly 
    # this is always not the first layer... so no extra condition there:D
    for ll in lc+1:number_of_layers
        number_of_tensors = TTNKit.n_tensors(net, ll)
        for pp in 1:number_of_tensors
            tn = ttn[ll,pp]
            #TODO            
        end        
    end
    
    res = norm(ttn[oc])
    return all(are_id), res
end

function Base.copy(ttn::TreeTensorNetwork)
    datac = deepcopy(ttn.data)
    ortho_centerc = deepcopy(ttn.ortho_center)
    netc = deepcopy(ttn.net)
    return TreeTensorNetwork(datac, ortho_centerc, netc)
end

include("./inner.jl")