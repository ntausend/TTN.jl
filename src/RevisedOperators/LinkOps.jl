function populate_physical_link_ops(net::BinaryNetwork,
                                    tpo::TPO_GPU)

    link_ops = Dict{Tuple{Tuple{Int,Int},Int}, Vector{Op_GPU}}()
    # Iterate over each node of first layer
    for n in eachindex(net,1)
        # Get child nodes of the first layer nodes
        childs = TTN.child_nodes(net, (1, n))
        # Iterate over each child node
        for (i, child) in enumerate(childs)
            link = ((1,n),i)  # layer, node, child_index
            ops_on_site = get_site_terms(tpo, child)
            # Store the operators on the link in the dictionary
            link_ops[link] = ops_on_site
        end
    end 
    return link_ops
end


function complete_contraction(net::BinaryNetwork,
                              ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
                              link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{Op_GPU}},
                              root::Tuple{Int,Int})
    
    tn = ttn0[root]
    collaps_list = Vector{ITensor}()
    bucket = get_id_terms(net, link_ops, root)
    ## get sites of ITensor
    tn_sites = extract_layer_node.(inds(tn))
    ## find index of open / parent connection to prime

    for (idd, ops) in bucket
        tn_p = tn
        tensor_list = [tn]
        for op in ops            
            op_site_id = findfirst(x -> x == op.site, tn_sites)
            op_site_ind = ind(tn_p, op_site_id)
            tn_p = prime(tn_p, op_site_ind)
            push!(tensor_list, op.op)
        end
        push!(tensor_list, dag(tn_p))
        opcon_seq = ITensors.optimal_contraction_sequence(tensor_list)
        tn_con = contract(tensor_list; sequence = opcon_seq)
        push!(collaps_list, tn_con)
    end
    tn_exp = sum(collaps_list)
    return tn_exp
end

complete_contraction(ptpo::ProjTPO_GPU, ttn0::TreeTensorNetwork) =
    complete_contraction(ptpo.net, ttn0, ptpo.link_ops, ptpo.ortho_center)

function upflow_to_root(net::BinaryNetwork,
                        ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
                        tpo::TPO_GPU,
                        root::Tuple{Int,Int};
                        use_gpu::Bool = false,
                        node_cache = Dict())

    link_ops = populate_physical_link_ops(net, tpo)
    # get path from top node to ortho_center = newroot
    top_node = (number_of_layers(net), 1)
    connect_path = pushfirst!(connecting_path(net, top_node, root), top_node)
    node_order = root == top_node ? NodeIterator(net) : reverse_bfs_nodes(net, root)

    for pos in node_order
        # stop at the new root itself
        if pos == root
            return link_ops
        end
        # determine the link on which to store your contracted ops
        if pos in connect_path
            next_node = next_on_path(connect_path, pos)
            # only valid because rerooting starts at the top node
            # -> only downflows from the top node
            link = (pos, which_child(net, next_node))
            coarse_ops = contract_ops(net, ttn0, link_ops, pos; open_link = next_node, use_gpu = use_gpu, node_cache = node_cache)
        else
            link = (parent_node(net, pos), which_child(net, pos))
            coarse_ops = contract_ops(net, ttn0, link_ops, pos; use_gpu = use_gpu, node_cache = node_cache)
        end
        
        link_ops[link] = coarse_ops
    end
    return link_ops
end

"""
    recalc_path_link_ops!(net, ttn0, link_ops, oldroot, newroot)

Recompute the link operators on the unique path that connects the original
root (the top node) of `net` to `newroot`.

The procedure is:
  1. Find the path between `top_node` and `newroot` (both ends included).
  2. Delete the stale link operators that live on that path from `link_ops`.
  3. Traverse the path **top‑down**, recomputing the coarse‑grained operator
     for each link that was just removed while keeping every off‑path operator
     untouched.

The implementation mirrors the logic of `upflow_rerooted` but is restricted to
the connecting path, reducing the run‑time from *O(|net|)* to
*O(length(path))*.
"""
function recalc_path_flows!(
        net::BinaryNetwork,
        ttn0::TreeTensorNetwork,
        link_ops::Dict,
        oldroot::Tuple{Int,Int},
        newroot::Tuple{Int,Int};
        use_gpu::Bool = false,
        node_cache = Dict()
    )

    # 1. Determine the unique path from the original root to the new root
    # top_node = (number_of_layers(net), 1)
    connect_path = pushfirst!(connecting_path(net, oldroot, newroot), oldroot)

    # 2. Remove the existing link operators that live on this path
    for pos in connect_path[1:end-1]          # skip the new root itself
        nxt   = next_on_path(connect_path, pos)
        if nxt[1] > pos[1]
            # going up the tree
            link = (nxt, which_child(net, pos))
        else
        # going down the tree
            link = (pos, which_child(net, nxt))
        end
        # println("Position: $pos, Next: $nxt, Link: $link")
        delete!(link_ops, link)   # silent if already absent
    end

    # 3. Recompute the link operators along the path (top‑down)
    for pos in connect_path[1:end-1]
        nxt  = next_on_path(connect_path, pos)
        if nxt[1] > pos[1] # layer of next node is greater than current node
            # going up the tree
            link = (nxt, which_child(net, pos))
            coarse_ops = contract_ops(net, ttn0, link_ops, pos; use_gpu = use_gpu, node_cache = node_cache)
        else
            # going down the tree
            link = (pos, which_child(net, nxt))
            coarse_ops = contract_ops(net, ttn0, link_ops, pos; open_link = nxt, use_gpu = use_gpu, node_cache = node_cache)
        end
        # println("Position: $pos, Next: $nxt, Link: $link")
        link_ops[link] = coarse_ops
    end

    return link_ops
end

function recalc_path_flows!(ptpo::ProjTPO_GPU,
                            ttn::TreeTensorNetwork,
                            newroot::Tuple{Int,Int};
                            use_gpu::Bool = false,
                            node_cache = Dict()
                            )

    recalc_path_flows!(ptpo.net, ttn, ptpo.link_ops, ptpo.ortho_center, newroot; use_gpu = use_gpu, node_cache = node_cache)
    ptpo.ortho_center = newroot                                     # keep state consistent
    return ptpo
end
function recalc_path_flows!(ptpo::ProjTPO_GPU,
                            ttn::TreeTensorNetwork,
                            oldroot::Tuple{Int,Int},
                            newroot::Tuple{Int,Int};
                            use_gpu::Bool = false,
                            node_cache = Dict()
                            )

    @assert oldroot == ptpo.ortho_center
    recalc_path_flows!(ptpo.net, ttn, ptpo.link_ops, oldroot, newroot; use_gpu = use_gpu, node_cache = node_cache)
    ptpo.ortho_center = newroot                                     # keep state consistent
    return ptpo
end

# function contract_ops(net::BinaryNetwork, ttn0::TreeTensorNetwork{BinaryNetwork{TTN.SimpleLattice{2, Index, Int64}}, ITensor},
#      link_ops::Dict{Tuple{Tuple{Int64, Int64}, Int64}, Vector{Op_GPU}}, pos::Tuple{Int,Int}; open_link::Tuple{Int,Int} = pos)

function contract_ops(net::BinaryNetwork,
                      ttn0::TreeTensorNetwork,
                      link_ops::Dict,
                      pos::Tuple{Int,Int};
                      open_link::Tuple{Int,Int} = pos,
                      use_gpu::Bool = false,
                      node_cache = Dict())

    op_vec = Vector{Op_GPU}()
    collaps_list = Vector{ITensor}()

    bucket = get_id_terms(net, link_ops, pos)

    if use_gpu
        tn_ = haskey(node_cache, pos) ? node_cache[pos] : gpu(ttn0[pos])
    else
        tn_ = ttn0[pos]
    end

    open_tag = link_tag(open_link...)
    open_index = findfirst(i -> hastags(i, open_tag), inds(tn_))
    if open_index === nothing
        error("Could not find open link index with tag $open_tag in node $pos")
    end

    tn_dag = dag(prime(tn_, ind(tn_, open_index)))

    for (idd, ops) in bucket
        tn_p = tn_dag
        tensor_list = [tn_]
        len_original = ops[1].original_length
        len_op = len_original
        len_con = length(ops)
        op_red = 0

        for op in ops
            op_tensor = use_gpu ? gpu(op.op) : op.op
            common_index = commonind(tn_, op_tensor)
            tn_p = prime(tn_p, common_index)
            push!(tensor_list, op_tensor)
            op_red += op_reduction(op)
        end

        push!(tensor_list, tn_p)
        opcon_seq = optimal_contraction_sequence(tensor_list)
        tn_con = contract(tensor_list; sequence = opcon_seq)

        if len_con > 1
            op_red += 1
        end
        len_op -= op_red


        if len_op > 1
            tn_con = cpu(tn_con)
            push!(op_vec, Op_GPU(idd, open_link, tn_con, len_original, len_op))
        else
            push!(collaps_list, tn_con)
        end
    end

    if !isempty(collaps_list)
        sum_collapse = sum(collaps_list)
        # sum_collapse = use_gpu ? cpu(sum_collapse) : sum_collapse
        push!(op_vec, Op_GPU(new_Op_GPU_id(), open_link, cpu(sum_collapse), 1, 1))
    end

    return op_vec
end

function extract_layer_node(index::Index)

    taglist = string.(collect(tags(index)))  # Vector{String}
    nl = nothing
    np = nothing
    for tag in taglist
        if startswith(tag, "nl=")
            nl = parse(Int, split(tag, "=")[2])
        elseif startswith(tag, "np=")
            np = parse(Int, split(tag, "=")[2])
        elseif startswith(tag, "n=")
            nl = 0
            np = parse(Int, split(tag, "=")[2])
        end
    end
    if isnothing(nl) || isnothing(np)
        error("Could not extract nl or np from index tags")
    end
    return nl, np
end

function link_tag(nl::Int, np::Int)
    if nl > 0
        return "Link,nl=$(nl),np=$(np)"
    else
        return "Site,SpinHalf,n=$(np)"
    end
end

function full_contraction(ttn::TreeTensorNetwork, tpo::TPO_GPU)
    ptpo = ProjTPO_GPU(ttn, tpo)
    return full_contraction(ttn, ptpo)
end

function full_contraction(ttn::TreeTensorNetwork, ptpo::ProjTPO_GPU)
    # set the ptpo to the correct position of the ttn
    ptpo = set_position!(ptpo, ttn)
    oc = ortho_center(ttn)

    # get the action of the operator on the orthogonlity center
    action = ∂A(ptpo, oc)
    T = ttn[oc]
    # build the contraction
    return dot(T, action(T))
end

function set_position!(pTPO::ProjTPO_GPU{N,T}, ttn::TreeTensorNetwork{N,T}; use_gpu::Bool = false, node_cache = Dict()) where {N,T}
    oc_projtpo = ortho_center(pTPO)
    oc_ttn     = ortho_center(ttn)
    # both structures should be gauged.. otherwise no real thing todo
    @assert !any(oc_ttn     .== -1)
    @assert !any(oc_projtpo .== -1)

    all(oc_projtpo .== oc_ttn) && return pTPO

    # move oc of link_operators from oc_projtpo to oc_ttn
    recalc_path_flows!(pTPO, ttn, oc_ttn; use_gpu = use_gpu, node_cache = node_cache)
    return pTPO
end

"""
    ∂A(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int})

Return a closure `action(T)` that applies the part of the *projected*
Hamiltonian living on node `pos` to an `ITensor T` on the **same** node.

For every operator–ID that still has support on one (or more) of the three
links attached to `pos` (parent, first-child, second-child) the routine

  1. collects the corresponding `ITensor`s that represent that operator’s
     pieces near `pos`;
  2. builds the minimal contraction network `[T, op₁, op₂, …]`;
  3. contracts it with an optimal sequence; and
  4. finally sums all such contributions.

This reproduces the behaviour of the original `∂A` but now talks to the new
`ProjTPO_GPU` data model.
"""

#=
function ∂A_GPU(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int}; use_gpu::Bool=false)
    net      = ptpo.net
    link_ops = ptpo.link_ops
    id_bucket = get_id_terms(net, link_ops, pos)

    envs = [map(g -> g.op, grp) for grp in values(id_bucket)]

    if use_gpu
        action = function (T::ITensor)
            isempty(envs) && return zero(T)

            T_gpu = gpu(T)
            @assert is_cu(T_gpu) "T is not on GPU"
            # @info "∂A_GPU: T storage" typeof(storage(T_gpu))

            acc_gpu = nothing
            for trm in envs
                ops_gpu = gpu(trm)
                # @info "∂A_GPU: op storage" typeof(storage(ops_gpu[1]))
                tensor_list = vcat(T_gpu, ops_gpu)
                seq = ITensors.optimal_contraction_sequence(tensor_list)
                contrib_gpu = noprime(contract(tensor_list; sequence = seq))
                acc_gpu === nothing ? (acc_gpu = contrib_gpu) : (acc_gpu += contrib_gpu)
            end
            return acc_gpu
        end
    else
        action = function (T::ITensor)
            isempty(envs) && return zero(T)

            acc = nothing
            for trm in envs
                tensor_list = vcat(T, trm)
                seq         = ITensors.optimal_contraction_sequence(tensor_list)
                contrib     = noprime(contract(tensor_list; sequence = seq))
                acc === nothing ? (acc = contrib) : (acc += contrib)
            end
            return acc
        end
    end

    return action
end
=#

# _∂A_impl wrapper for type-stability
function ∂A_GPU(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int}; use_gpu::Bool=false)
    return use_gpu ? _∂A_impl(ptpo, pos, Val(:gpu)) : _∂A_impl(ptpo, pos, Val(:cpu))
end

function _∂A_impl(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int}, ::Val{:gpu})
    net      = ptpo.net
    link_ops = ptpo.link_ops
    id_bucket = get_id_terms(net, link_ops, pos)
    envs = [map(g -> g.op, grp) for grp in values(id_bucket)]

    return function (T::ITensor)
        isempty(envs) && return zero(T)

        T_gpu = gpu(T)
        @assert is_cu(T_gpu) "T is not on GPU"

        acc_gpu = ITensor(inds(T)...)
        for trm in envs
            ops_gpu = gpu.(trm)
            tensor_list = vcat(T_gpu, ops_gpu)
            seq = ITensors.optimal_contraction_sequence(tensor_list)
            contrib_gpu = noprime(contract(tensor_list; sequence = seq))
            acc_gpu += contrib_gpu
        end
        return acc_gpu
    end
end

function _∂A_impl(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int}, ::Val{:cpu})
    net      = ptpo.net
    link_ops = ptpo.link_ops
    id_bucket = get_id_terms(net, link_ops, pos)
    envs = [map(g -> g.op, grp) for grp in values(id_bucket)]

    return function (T::ITensor)
        isempty(envs) && return zero(T)

        # acc = ITensor(inds(T)...)
        acc = nothing
        for trm in envs
            tensor_list = vcat(T, trm)
            seq         = ITensors.optimal_contraction_sequence(tensor_list)
            contrib     = noprime(contract(tensor_list; sequence = seq))
            # acc += contrib
            acc === nothing ? (acc = contrib) : (acc += contrib)
        end
        return acc
    end
end

function ∂A2_GPU(ptpo::ProjTPO_GPU, isom::ITensor, pos::Tuple{Int,Int}; use_gpu::Bool=false)
    return use_gpu ? _∂A2_impl(ptpo, isom, pos, Val(:gpu)) : _∂A2_impl(ptpo, isom, pos, Val(:cpu))
end

function _∂A2_impl(ptpo::ProjTPO_GPU, isom::ITensor, pos::Tuple{Int,Int}, ::Val{:gpu})

    id_bucket = get_id_terms(ptpo.net, ptpo.link_ops, pos)

    function action(link::ITensor)

        acc = ITensor(inds(link)...)

        isom = gpu(isom)
        link = gpu(link)

        for (_, ops) in id_bucket

            tensor_list = ITensor[]
            push!(tensor_list, isom, link)

            isom_p = prime(isom, commonind(isom, link))
            
            for op in ops
                op_tensor = gpu(op.op)
                push!(tensor_list, op_tensor)

                common_index = commonind(op_tensor, isom_p)
                if !isnothing(common_index)
                    isom_p = prime(isom_p, common_index)
                end
            end

            push!(tensor_list, dag(isom_p))

            seq = optimal_contraction_sequence(tensor_list)
            contrib = noprime(contract(tensor_list; sequence = seq))

            acc += contrib
        end
        return acc
    end
    return action
end

function _∂A2_impl(ptpo::ProjTPO_GPU, isom::ITensor, pos::Tuple{Int,Int}, ::Val{:cpu})
    
    id_bucket = get_id_terms(ptpo.net, ptpo.link_ops, pos)

    function action(link::ITensor)

        # acc = ITensor(inds(link)...)
        acc = nothing

        for (_, ops) in id_bucket

            tensor_list = ITensor[]
            push!(tensor_list, isom, link)

            isom_p = prime(isom, commonind(isom, link))
            
            for op in ops
                op_tensor = op.op
                push!(tensor_list, op_tensor)

                common_index = commonind(op_tensor, isom_p)
                if !isnothing(common_index)
                    isom_p = prime(isom_p, common_index)
                end
            end

            push!(tensor_list, dag(isom_p))

            seq = optimal_contraction_sequence(tensor_list)
            contrib = noprime(contract(tensor_list; sequence = seq))

            # acc += contrib
            acc === nothing ? (acc = contrib) : (acc += contrib)
        end
        return acc
    end
    return action
end