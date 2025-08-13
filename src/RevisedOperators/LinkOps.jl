function populate_physical_link_ops(net::BinaryNetwork, tpo::TPO_GPU)

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

function upflow_to_root(net::BinaryNetwork, ttn0::TreeTensorNetwork, tpo::TPO_GPU, root::Tuple{Int,Int};
                        use_gpu::Bool = false, node_cache = Dict())

    link_ops = populate_physical_link_ops(net, tpo)
    # get path from top node to new root = ortho_center
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

function upflow_root_threads(net::BinaryNetwork, ttn0::TreeTensorNetwork, tpo::TPO_GPU, root::Tuple{Int,Int};
                        use_gpu::Bool = false, node_cache = Dict())

    link_ops = populate_physical_link_ops(net, tpo)
    # get path from top node to new root = ortho_center
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
            coarse_ops = contract_ops_threads(net, ttn0, link_ops, pos; open_link = next_node, use_gpu = use_gpu, node_cache = node_cache)
        else
            link = (parent_node(net, pos), which_child(net, pos))
            coarse_ops = contract_ops_threads(net, ttn0, link_ops, pos; use_gpu = use_gpu, node_cache = node_cache)
        end
        
        link_ops[link] = coarse_ops
    end
    return link_ops
end

function upflow_root_spawn(net::BinaryNetwork, ttn0::TreeTensorNetwork, tpo::TPO_GPU, root::Tuple{Int,Int};
                        use_gpu::Bool = false, node_cache = Dict())

    link_ops = populate_physical_link_ops(net, tpo)
    # get path from top node to new root = ortho_center
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
            coarse_ops = contract_ops_spawn(net, ttn0, link_ops, pos; open_link = next_node, use_gpu = use_gpu, node_cache = node_cache)
        else
            link = (parent_node(net, pos), which_child(net, pos))
            coarse_ops = contract_ops_spawn(net, ttn0, link_ops, pos; use_gpu = use_gpu, node_cache = node_cache)
        end
        
        link_ops[link] = coarse_ops
    end
    return link_ops
end

"""
    recalc_path_link_ops!(net, ttn0, link_ops, oldroot, newroot)

Recompute the link operators on the unique path that connects the original
root (the top node) of `net` to `newroot`.
  1. Find the path between `top_node` and `newroot` (both ends included).
  2. Delete the stale link operators that live on that path from `link_ops`.
  3. Traverse the path **top‑down**, recomputing the coarse‑grained operator
     for each link that was just removed while keeping every off‑path operator
     untouched.
"""

function recalc_path_flows!(net::BinaryNetwork, ttn0::TreeTensorNetwork, link_ops::Dict,
                            oldroot::Tuple{Int,Int}, newroot::Tuple{Int,Int};
                            use_gpu::Bool = false, node_cache = Dict())

    oldroot == newroot && return link_ops

    path = pushfirst!(connecting_path(net, oldroot, newroot), oldroot)

    link_between(pos, nxt) = nxt[1] > pos[1] ? (nxt, which_child(net, pos)) : (pos, which_child(net, nxt))

    # Remove stale operators
    for i in 1:length(path)-1
        delete!(link_ops, link_between(path[i], path[i+1]))
    end

    # Recompute fresh operators
    for i in 1:length(path)-1
        pos, nxt = path[i], path[i+1]
        link = link_between(pos, nxt)

        coarse_ops = nxt[1] > pos[1] ?
            contract_ops(net, ttn0, link_ops, pos; use_gpu = use_gpu, node_cache = node_cache) :
            contract_ops(net, ttn0, link_ops, pos; open_link = nxt, use_gpu = use_gpu, node_cache = node_cache)

        link_ops[link] = coarse_ops
    end

    return link_ops
end

function recalc_path_flows!(ptpo::ProjTPO_GPU, ttn::TreeTensorNetwork, newroot::Tuple{Int,Int};
                            use_gpu::Bool = false, node_cache = Dict())

    recalc_path_flows!(ptpo.net, ttn, ptpo.link_ops, ptpo.ortho_center, newroot; use_gpu = use_gpu, node_cache = node_cache)
    ptpo.ortho_center = newroot
    return ptpo
end
function recalc_path_flows!(ptpo::ProjTPO_GPU, ttn::TreeTensorNetwork,
                            oldroot::Tuple{Int,Int}, newroot::Tuple{Int,Int};
                            use_gpu::Bool = false, node_cache = Dict())

    @assert oldroot == ptpo.ortho_center
    recalc_path_flows!(ptpo.net, ttn, ptpo.link_ops, oldroot, newroot; use_gpu = use_gpu, node_cache = node_cache)
    ptpo.ortho_center = newroot
    return ptpo
end

# Original version of contract_ops

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

        # seq = length(tensor_list) > 3 ? [[[1,2],3],4] : [[1,2],3] # op first, then tn_p should always be optimal
        # @assert sum(ITensors.contraction_cost(tensor_list; sequence = seq)) == sum(ITensors.contraction_cost(tensor_list; sequence = optimal_contraction_sequence(tensor_list))) "manual sequence does not match optimal sequence"
        # opt_seq = optimal_contraction_sequence(tensor_list)
        # default left to right is optimal
        tn_con = contract(tensor_list)

        # reduce op length by previous reductions and current contraction
        len_op = len_original - op_red - (len_con > 1 ? 1 : 0)

        if len_op > 1
            tn_con = cpu(tn_con)
            push!(op_vec, Op_GPU(idd, open_link, tn_con, len_original, len_op))
        else
            push!(collaps_list, tn_con)
        end
    end

    if !isempty(collaps_list)
        sum_collapse = sum(collaps_list)
        push!(op_vec, Op_GPU(new_Op_GPU_id(), open_link, cpu(sum_collapse), 1, 1))
    end

    return op_vec
end

# Threads parallelized version
function contract_ops_threads(net::BinaryNetwork,
                              ttn0::TreeTensorNetwork,
                              link_ops::Dict,
                              pos::Tuple{Int,Int};
                              open_link::Tuple{Int,Int} = pos,
                              use_gpu::Bool = false,
                              node_cache = Dict())

    # 1) Build bucket and load center tensor
    bucket = get_id_terms(net, link_ops, pos)
    tn0 = ttn0[pos]
    if use_gpu && haskey(node_cache, pos)
        tn0 = gpu(node_cache[pos])
    elseif use_gpu
        tn0 = gpu(tn0)
    end

    # 2) Prepare the primed DAG
    open_tag = link_tag(open_link...)
    idx = findfirst(i -> hastags(i, open_tag), inds(tn0))
    @assert idx !== nothing "Could not find open link $open_tag"
    tn_dag_base = dag(prime(tn0, ind(tn0, idx)))

    # 3) Thread-local result buffers
    ids         = collect(keys(bucket))
    N           = length(ids)
    ops_per_th  = Vector{Vector{Op_GPU}}(undef, N)
    coll_per_th = Vector{Vector{ITensor}}(undef, N)

    # 4) Parallel loop
    @threads for ti in 1:N
        idd = ids[ti]
        ops = bucket[idd]

        local_ops  = Op_GPU[]
        local_coll = ITensor[]

        # copy inputs
        tn     = tn0
        tn_dag = tn_dag_base
        len_orig = ops[1].original_length
        total_red = 0

        # build contraction list
        T = ITensor[tn]
        for op in ops
            t_op = use_gpu ? gpu(op.op) : op.op
            ci   = commonind(tn, t_op)
            tn_dag = prime(tn_dag, ci)
            push!(T, t_op)
            total_red += op_reduction(op)
        end
        push!(T, tn_dag)

        # do the contraction
        tn_con = contract(T)

        # compute new length
        len_contr = length(ops) > 1 ? 1 : 0
        len_new   = len_orig - total_red - len_contr

        if len_new > 1
            tn_con = cpu(tn_con)
            push!(local_ops, Op_GPU(idd, open_link, tn_con, len_orig, len_new))
        else
            push!(local_coll, tn_con)
        end

        ops_per_th[ti]   = local_ops
        coll_per_th[ti]  = local_coll
    end

    # 5) Gather and finalize
    op_vec   = vcat(ops_per_th...)
    all_coll = vcat(coll_per_th...)

    # proper GPU sync
    if use_gpu
        CUDA.synchronize()
    end

    if !isempty(all_coll)
        s = sum(all_coll)
        push!(op_vec, Op_GPU(new_Op_GPU_id(), open_link, cpu(s), 1, 1))
    end

    return op_vec
end

# Spawn parallelized version
function contract_ops_spawn(net::BinaryNetwork,
                            ttn0::TreeTensorNetwork,
                            link_ops::Dict,
                            pos::Tuple{Int,Int};
                            open_link::Tuple{Int,Int} = pos,
                            use_gpu::Bool = false,
                            node_cache = Dict())

    # 1) Build bucket and load center tensor
    bucket = get_id_terms(net, link_ops, pos)
    tn0 = ttn0[pos]
    if use_gpu && haskey(node_cache, pos)
        tn0 = gpu(node_cache[pos])
    elseif use_gpu
        tn0 = gpu(tn0)
    end

    # 2) Prepare the primed DAG
    open_tag = link_tag(open_link...)
    idx = findfirst(i -> hastags(i, open_tag), inds(tn0))
    @assert idx !== nothing "Could not find open link $open_tag"
    tn_dag_base = dag(prime(tn0, ind(tn0, idx)))

    # 3) Spawn one Task per bucket entry
    futures = Dict{Int,Task}()
    for (idd, ops) in bucket
        futures[idd] = @spawn begin
            local_ops  = Op_GPU[]
            local_coll = ITensor[]

            tn     = tn0
            tn_dag = tn_dag_base
            len_orig  = ops[1].original_length
            total_red = 0

            T = ITensor[tn]
            for op in ops
                t_op = use_gpu ? gpu(op.op) : op.op
                ci   = commonind(tn, t_op)
                tn_dag = prime(tn_dag, ci)
                push!(T, t_op)
                total_red += op_reduction(op)
            end
            push!(T, tn_dag)

            tn_con = contract(T)

            len_contr = length(ops) > 1 ? 1 : 0
            len_new   = len_orig - total_red - len_contr

            if len_new > 1
                push!(local_ops, Op_GPU(idd, open_link, cpu(tn_con), len_orig, len_new))
            else
                push!(local_coll, tn_con)
            end

            return (ops = local_ops, coll = local_coll)
        end
    end

    # 4) Fetch results and finalize
    op_vec   = Op_GPU[]
    all_coll = ITensor[]
    for fut in values(futures)
        res = fetch(fut)
        append!(op_vec, res.ops)
        append!(all_coll, res.coll)
    end

     # proper GPU sync
    if use_gpu
        CUDA.synchronize()
    end

    if !isempty(all_coll)
        s = sum(all_coll)
        push!(op_vec, Op_GPU(new_Op_GPU_id(), open_link, cpu(s), 1, 1))
    end

    return op_vec
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

# _∂A_impl wrapper for type-stability
function ∂A_GPU(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int}; use_gpu::Bool=false)
    return use_gpu ? _∂A_impl(ptpo, pos, Val(:gpu)) : _∂A_impl(ptpo, pos, Val(:cpu))
end


# original with manual contraction sequence
function _∂A_impl(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int}, ::Val{:gpu})
    net      = ptpo.net
    link_ops = ptpo.link_ops
    id_bucket = get_id_terms(net, link_ops, pos)
    # envs = [map(g -> g.op, grp) for grp in values(id_bucket)]
    envs = [gpu.([g.op for g in grp]) for grp in values(id_bucket)]


    return function (T::ITensor)
        isempty(envs) && return zero(T)

        T_gpu = gpu(T)

        acc_gpu = ITensor(inds(T)...)
        for ops_gpu in envs
            tensor_list = vcat(T_gpu, ops_gpu)
            # seq = ITensors.optimal_contraction_sequence(tensor_list)
            # left to right contraction is optimal
            contrib_gpu = noprime(contract(tensor_list))
            acc_gpu += contrib_gpu
        end
        return acc_gpu
    end
end


#=
# original with manual contraction sequence - parallelized (memory issues)
function _∂A_impl(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int}, ::Val{:gpu})
    net      = ptpo.net
    link_ops = ptpo.link_ops
    id_bucket = get_id_terms(net, link_ops, pos)
    envs = [map(g -> g.op, grp) for grp in values(id_bucket)]

    return function (T::ITensor)
        isempty(envs) && return zero(T)

        T_gpu = gpu(T)

        # Preallocate vector for tasks
        tasks = Vector{Task}(undef, length(envs))

        for i in eachindex(envs)
            trm = envs[i]
            tasks[i] = @spawn begin
                ops_gpu = gpu.(trm)
                tensor_list = vcat(T_gpu, ops_gpu)
                contrib_gpu = noprime(contract(tensor_list))  # manual left-to-right contraction
                return contrib_gpu
            end
        end

        # Accumulate results
        acc_gpu = ITensor(inds(T)...)
        for task in tasks
            acc_gpu += fetch(task)
        end

        return acc_gpu
    end
end
=#

#=
# map and manual contraction sequence version
function _∂A_impl(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int}, ::Val{:gpu})
    net = ptpo.net
    link_ops = ptpo.link_ops
    id_bucket = get_id_terms(net, link_ops, pos)

    # Preload operators to GPU
    envs = [gpu.([g.op for g in grp]) for grp in values(id_bucket)]

    return function (T_gpu::ITensor)
        # isempty(envs) && return zero(T)
        @assert is_cu(T_gpu) "Tensor should already be on GPU"

        # T_gpu = gpu(T)

        contributions = map(envs) do ops_gpu
            tensors = (T_gpu, ops_gpu...)
            # seq = optimal_contraction_sequence(tensors)
            # left to right contraction is optimal
            noprime(contract(tensors))
        end

        # return isempty(contributions) ? zero(T_gpu) : reduce(+, contributions)
        return reduce(+, contributions)
    end
end
=#

## write similar to the gpu version
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
            # seq         = ITensors.optimal_contraction_sequence(tensor_list)
            # left to right contraction is optimal
            contrib     = noprime(contract(tensor_list))
            # acc += contrib
            acc === nothing ? (acc = contrib) : (acc += contrib)
        end
        return acc
    end
end

# _∂A2_impl wrapper for type-stability
function ∂A2_GPU(ptpo::ProjTPO_GPU, isom::ITensor, pos::Tuple{Int,Int}; use_gpu::Bool=false)
    return use_gpu ? _∂A2_impl(ptpo, isom, pos, Val(:gpu)) : _∂A2_impl(ptpo, isom, pos, Val(:cpu))
end

#=
# manual contraction sequence version
function _∂A2_impl(ptpo::ProjTPO_GPU, isom::ITensor, pos::Tuple{Int,Int}, ::Val{:gpu})
    id_bucket = get_id_terms(ptpo.net, ptpo.link_ops, pos)
    isom_gpu = gpu(isom)

    function action(link::ITensor)
        acc = ITensor(inds(link)...)  # preallocate accumulator

        link_gpu = gpu(link)

        for (_, ops) in id_bucket
            isom_ops = ITensor[]
            link_ops = ITensor[]

            isom_p = prime(isom_gpu, commonind(isom_gpu, link_gpu))

            for op in ops
                op_tensor = gpu(op.op)
                idx = commonind(op_tensor, isom_p)
                if !isnothing(idx)
                    isom_p = prime(isom_p, idx)
                    push!(isom_ops, op_tensor)
                else
                    push!(link_ops, op_tensor)
                end
            end

            # Contract isometry leg first (including dag(isom_p))
            isom_con = isempty(isom_ops) ? isom_gpu * dag(isom_p) : contract([isom_gpu, isom_ops..., dag(isom_p)])

            # Contract link leg
            link_con = isempty(link_ops) ? link_gpu : link_gpu * link_ops[1]

            # Combine both
            contrib = noprime(isom_con * link_con)
            acc += contrib
        end

        return acc
    end

    return action
end
=#

# preloading operators
function _∂A2_impl(ptpo::ProjTPO_GPU, isom::ITensor, pos::Tuple{Int,Int}, ::Val{:gpu})
    # Bucket by operator id near `pos`
    id_bucket = get_id_terms(ptpo.net, ptpo.link_ops, pos)  # groups terms by id
    isom_gpu = gpu(isom)  # fixed for this closure

    # Preload + preclassify once per operator-id:
    #  - iso_ops: ops that share an index with `isom`
    #  - link_ops: the rest
    #  - prime_counts: how many extra primes each isom index needs
    pre_envs = Vector{Tuple{Vector{ITensor}, Vector{ITensor}, Dict{TagSet,Int}}}()
    for (_, ops) in id_bucket
        iso_ops  = ITensor[]
        lnk_ops  = ITensor[]
        pcounts  = Dict{TagSet,Int}()

        for op in ops
            og = gpu(op.op)  # preload op tensor to GPU
            # classify wrt the *base* isometry (independent of the link argument)
            if (sh = commonind(og, isom_gpu)) !== nothing
                push!(iso_ops, og)
                tg = tags(sh)
                pcounts[tg] = get(pcounts, tg, 0) + 1  # prime this index once per touching op
            else
                push!(lnk_ops, og)
            end
        end
        push!(pre_envs, (iso_ops, lnk_ops, pcounts))
    end

    # The closure used by the Krylov routine
    return function (link_gpu::ITensor)
        # link_gpu = gpu(link)
        @assert is_cu(link_gpu) "Link should already be on GPU"

        acc = nothing
        for (iso_ops, lnk_ops, pcounts) in pre_envs
            # prime the isometry by the open-link index (depends on `link`)
            isom_p = prime(isom_gpu, commonind(isom_gpu, link_gpu))
            # then apply the additional primes required by the ops on isom-legs
            for (tg, n) in pcounts
                @inbounds for _ = 1:n
                    isom_p = prime(isom_p, tg)
                end
            end

            # contract isometry side: (isom ・ ops_on_isom ・ dag(isom_p))
            iso_list  = ITensor[isom_gpu; iso_ops...; dag(isom_p)]
            iso_con   = contract(iso_list)

            # contract link side: (link ・ ops_on_link)
            link_list = ITensor[link_gpu; lnk_ops...]
            link_con  = contract(link_list)

            contrib = noprime(iso_con * link_con)
            acc === nothing ? (acc = contrib) : (acc += contrib)
        end
        return acc === nothing ? zero(link_gpu) : acc
    end
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

# Helper function to create a link tag
function link_tag(nl::Int, np::Int)
    if nl > 0
        return "Link,nl=$(nl),np=$(np)"
    else
        return "Site,SpinHalf,n=$(np)"
    end
end
