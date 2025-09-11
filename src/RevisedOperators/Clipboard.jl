## map version of contract_ops
function contract_ops(net::BinaryNetwork,
                      ttn0::TreeTensorNetwork,
                      link_ops::Dict,
                      pos::Tuple{Int,Int};
                      open_link::Tuple{Int,Int} = pos,
                      use_gpu::Bool = false,
                      node_cache = Dict())

    bucket = get_id_terms(net, link_ops, pos)

    tn_ = use_gpu ? get(node_cache, pos, gpu(ttn0[pos])) : ttn0[pos]

    open_tag = link_tag(open_link...)
    open_index = findfirst(i -> hastags(i, open_tag), inds(tn_))
    open_index === nothing && error("Could not find open link index with tag $open_tag in node $pos")

    tn_dag = dag(prime(tn_, ind(tn_, open_index)))

    results = map(collect(bucket)) do (idd, ops)
        tensor_list = [tn_]
        tn_p = tn_dag
        op_red = 0

        tensor_list_ops = map(ops) do op
            op_tensor = use_gpu ? gpu(op.op) : op.op
            tn_p = prime(tn_p, commonind(tn_, op_tensor))
            op_red += op_reduction(op)
            op_tensor
        end

        full_list = vcat(tensor_list, tensor_list_ops, [tn_p])
        tn_con = contract(full_list; sequence = optimal_contraction_sequence(full_list))

        len_original = ops[1].original_length
        len_op = len_original - op_red - (length(ops) > 1 ? 1 : 0)

        if len_op > 1
            Op_GPU(idd, open_link, cpu(tn_con), len_original, len_op)
        else
            tn_con  # collect this separately
        end
    end

    # Separate Op_GPU from collapsed ITensor contributions
    op_vec = [res for res in results if res isa Op_GPU]
    collaps_list = [res for res in results if res isa ITensor]

    if !isempty(collaps_list)
        sum_collapse = sum(collaps_list)
        push!(op_vec, Op_GPU(new_Op_GPU_id(), open_link, cpu(sum_collapse), 1, 1))
    end

    return op_vec
end

# batched version of contract_ops
function contract_ops(net::BinaryNetwork,
                                      ttn0::TreeTensorNetwork,
                                      link_ops::Dict,
                                      pos::Tuple{Int,Int};
                                      open_link::Tuple{Int,Int} = pos,
                                      use_gpu::Bool = false,
                                      node_cache = Dict(),)

    bucket = get_id_terms(net, link_ops, pos)
    tn_ = use_gpu ? get(node_cache, pos, gpu(ttn0[pos])) : ttn0[pos]

    open_tag = link_tag(open_link...)
    open_index = findfirst(i -> hastags(i, open_tag), inds(tn_))
    open_index === nothing && error("Could not find open link index")

    tn_dag = dag(prime(tn_, ind(tn_, open_index)))

    op_vec = Op_GPU[]
    small_terms = ITensor[]

    for (idd, ops) in bucket
        op_red = sum(op_reduction, ops)
        op_tensors = gpu.([op.op for op in ops])
        tn_p = reduce((tp, op) -> prime(tp, commonind(tn_, op)), op_tensors; init = tn_dag)

        tensors = (tn_, op_tensors..., tn_p)
        seq = optimal_contraction_sequence(tensors)
        tn_con = contract(tensors; sequence = seq)

        len_original = ops[1].original_length
        len_op = len_original - op_red - (length(ops) > 1 ? 1 : 0)

        if len_op > 1
            push!(op_vec, Op_GPU(idd, open_link, cpu(tn_con), len_original, len_op))
        else
            push!(small_terms, tn_con)
        end
    end

    if !isempty(small_terms)
        sum_collapse = reduce(+, small_terms)
        push!(op_vec, Op_GPU(new_Op_GPU_id(), open_link, cpu(sum_collapse), 1, 1))
    end

    return op_vec
end

# Original _∂A_impl for gpu

function _∂A_impl(ptpo::ProjTPO_GPU, pos::Tuple{Int,Int}, ::Val{:gpu})
    net      = ptpo.net
    link_ops = ptpo.link_ops
    id_bucket = get_id_terms(net, link_ops, pos)
    envs = [map(g -> g.op, grp) for grp in values(id_bucket)]

    return function (T::ITensor)
        isempty(envs) && return zero(T)

        T_gpu = gpu(T)

        acc_gpu = ITensor(inds(T)...)
        for trm in envs
            ops_gpu = gpu.(trm)
            tensor_list = vcat(T_gpu, ops_gpu)
            # seq = ITensors.optimal_contraction_sequence(tensor_list)
            # left to right contraction is optimal
            contrib_gpu = noprime(contract(tensor_list))
            acc_gpu += contrib_gpu
        end
        return acc_gpu
    end
end


# Original _∂A2_impl for gpu
function _∂A2_impl(ptpo::ProjTPO_GPU, isom::ITensor, pos::Tuple{Int,Int}, ::Val{:gpu})

    id_bucket = get_id_terms(ptpo.net, ptpo.link_ops, pos)

    function action(link::ITensor)

        acc = ITensor(inds(link)...)

        isom = gpu(isom)
        link = gpu(link)

        for (_, ops) in id_bucket

            tensor_list = ITensor[]
            push!(tensor_list, isom, link)
            # tensor_list = vcat(isom, link)

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

# contraction sequence debugging version _∂A2_impl for gpu
function _∂A2_impl(ptpo::ProjTPO_GPU, isom::ITensor, pos::Tuple{Int,Int}, ::Val{:gpu})

    id_bucket = get_id_terms(ptpo.net, ptpo.link_ops, pos)

    function action(link::ITensor)

        acc = ITensor(inds(link)...)

        isom = gpu(isom)
        link = gpu(link)

        for (_, ops) in id_bucket

            tensor_list = ITensor[]
            push!(tensor_list, isom, link)

            isom_list = ITensor[]
            push!(isom_list, isom)
            link_list = ITensor[]
            push!(link_list, link)

            isom_p = prime(isom, commonind(isom, link))

            for op in ops
                op_tensor = gpu(op.op)
                push!(tensor_list, op_tensor)

                common_index = commonind(op_tensor, isom_p)
                if !isnothing(common_index)
                    isom_p = prime(isom_p, common_index)
                    push!(isom_list, op_tensor)
                else
                    push!(link_list, op_tensor)
                end
            end

            push!(isom_list, dag(isom_p))
            push!(tensor_list, dag(isom_p))

            seq = optimal_contraction_sequence(tensor_list)
            cost_opt = sum(ITensors.contraction_cost(tensor_list; sequence = seq))
            # contrib = noprime(contract(tensor_list; sequence = seq))

            isom_con = contract(isom_list)
            link_con = length(link_list) > 1 ? contract(link_list) : link

            contrib = noprime(contract(isom_con, link_con))

            cost_man = sum(ITensors.contraction_cost(isom_list)) + (length(link_list) > 1 ? sum(ITensors.contraction_cost(link_list)) : 0) + sum(ITensors.contraction_cost([isom_con, link_con]))
            @assert cost_opt == cost_man
            acc += contrib
        end
        return acc
    end
    return action
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