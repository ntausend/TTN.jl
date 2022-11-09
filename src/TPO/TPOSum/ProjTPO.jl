struct ProjTensorProductOperator{D, S<:IndexSpace, I<:Sector}
    net::AbstractNetwork{D, S, I}
    tpo::AbstractTensorProductOperator

    bottom_envs::Vector{Vector{Vector{AbstractTensorMap}}}
    top_envs::Vector{Vector{AbstractTensorMap}}
 
    Indices::Dict
end


function contract_tensors(tensorList::Vector{AbstractTensorMap}, indexList::Vector{Vector{Int}})
    unique_indices = Any[]
    double_indices = Any[]
    flatIndexList = vcat(indexList...)  

    while !(isempty(flatIndexList))
        el = popfirst!(flatIndexList)
        if !(el in flatIndexList) && !(el in double_indices)
            append!(unique_indices, [el])
        else
            append!(double_indices, [el])
        end
    end

    n = -1
    for unique_ind in unique_indices
        for list in indexList
            if unique_ind in list
                replace!(list, unique_ind => n)
                n -= 1
                break
            end
        end
    end 

    return (unique_indices, @ncon(tensorList, indexList))
end


function linear_index(net::AbstractNetwork, pos::Tuple{Int, Int})

    pos[1] == 0 && return pos[2]

    n_count = TTNKit.number_of_sites(net)

    for ll in 1:pos[1]-1
        n_count += TTNKit.number_of_tensors(net, ll)
    end

    return pos[2] + n_count
end

function build_Dict(net::AbstractNetwork)

    n_sites = TTNKit.number_of_sites(net)
    n_tensors = TTNKit.number_of_tensors(net) + n_sites

    indices = Dict()

    n_chds = 0
    for pp in 1:n_sites
        for n in 1:TTNKit.number_of_child_nodes(net, (1,pp))
            indices[("bottom", 1, pp, n)] = [n_chds + n, n_chds + n + n_tensors, n_chds + n + 2*n_tensors, 1 + n_chds + n + 2*n_tensors]
        end
        n_chds += TTNKit.number_of_child_nodes(net, (1,pp))
    end

    indices[(0, 0)] = [1+2*n_tensors]
    indices[(0, 1)] = [1+n_sites+2*n_tensors]
    
    for ll in TTNKit.eachlayer(net)
        for pp in TTNKit.eachindex(net, ll)
            indices[(ll, pp)] = append!([linear_index(net, pos_chd) for pos_chd in TTNKit.child_nodes(net, (ll,pp))], [linear_index(net, (ll,pp))])
            indices[(-ll, pp)] = append!([linear_index(net, (ll,pp)) + n_tensors], [linear_index(net, pos_chd)+ n_tensors for pos_chd in TTNKit.child_nodes(net, (ll,pp))])
        end
    end

    indices[("top", TTNKit.number_of_layers(net), 1)] = [n_tensors, 2*n_tensors, 2*n_tensors + n_sites + 2, 2*n_tensors + n_sites + 3]
    indices[(TTNKit.number_of_layers(net)+1, 1)] = [2*n_tensors + n_sites + 2]
    indices[(-TTNKit.number_of_layers(net)-1, 1)] = [2*n_tensors + n_sites + 3]


    return indices
end

function bottom_env(ttn::TreeTensorNetwork, Indices::Dict, tpo::AbstractTensorProductOperator)
    
    net = TTNKit.network(ttn)
    indices = copy(Indices)

    # first two vectors are for layer and position within the layer respectivley, 
    # third one is for enumerating the bottom envs - one for each child leg

    bottom_envs = Vector{Vector{Vector{AbstractTensorMap}}}(undef, TTNKit.number_of_layers(net)) 
    bottom_envs[1] = Vector{Vector{AbstractTensorMap}}(undef, TTNKit.number_of_tensors(net, 1))

    n_total = 0

    for pp in TTNKit.eachindex(net, 1)
        n_chds = TTNKit.number_of_child_nodes(net, (1,pp))
        bottom_envs[1][pp] = Vector{AbstractTensorMap}([tpo.data[n] for n in 1+n_total:n_total+n_chds])
        n_total += n_chds
    end

    for ll in Iterators.drop(TTNKit.eachlayer(net), 1)
        bottom_envs[ll] = Vector{Vector{AbstractTensorMap}}(undef, TTNKit.number_of_tensors(net, ll))

        for pp in TTNKit.eachindex(net, ll)
            n_chds = TTNKit.number_of_child_nodes(net, (ll,pp))
            bottom_envs[ll][pp] = Vector{AbstractTensorMap}(undef, n_chds)

            for chd_nd in TTNKit.child_nodes(net, (ll,pp))
                chd_idx = TTNKit.index_of_child(net, chd_nd)

                tensorList = Vector{AbstractTensorMap}([copy(ttn[(chd_nd)]), copy(adjoint(ttn[(chd_nd)]))])
                indexList = [copy(indices[chd_nd]), copy(indices[(-chd_nd[1], chd_nd[2])])]

                for chd_chd_nd in TTNKit.child_nodes(net, chd_nd)
                    chd_chd_idx = TTNKit.index_of_child(net, chd_chd_nd)
                    append!(tensorList, [copy(bottom_envs[chd_nd[1]][chd_nd[2]][chd_chd_idx])])
                    append!(indexList, [copy(indices[("bottom", chd_nd[1], chd_nd[2], chd_chd_idx)])])
                end
                
                (newIndices, bottom_envs[ll][pp][chd_idx]) = contract_tensors(tensorList, indexList)
                indices[("bottom", ll, pp, chd_idx)] = newIndices 
            end        
        end
    end 

    return indices, bottom_envs
end

function top_env(ttn::TreeTensorNetwork{D}, indices::Dict, bottom_envs::Vector{Vector{Vector{AbstractTensorMap}}}) where {D}
    
    net = TTNKit.network(ttn)

    # first two vectors are for layer and position within the layer respectivley
    top_envs = Vector{Vector{AbstractTensorMap}}(undef, TTNKit.number_of_layers(net))

    # identity matrix for the top environment of the top node
    top_envs[TTNKit.number_of_layers(net)] = [TensorKit.id(domain(ttn[(TTNKit.number_of_layers(net), 1)]))⊗TensorKit.id(domain(ttn[(TTNKit.number_of_layers(net), 1)])')]

    for ll in Iterators.drop(reverse(TTNKit.eachlayer(net)), 1)
        top_envs[ll] = Vector{AbstractTensorMap}(undef, TTNKit.number_of_tensors(net, ll))

        for pp in TTNKit.eachindex(net, ll)
            prnt_nd = TTNKit.parent_node(net, (ll,pp))
            idx_chd = TTNKit.index_of_child(net, (ll,pp))


            # top environment of node (ll,pp) is built by contracting the parent node with its top environment and its remaining bottom environments
            tensorList = Vector{AbstractTensorMap}([copy(ttn[(prnt_nd)]), copy(adjoint(ttn[(prnt_nd)])), copy(top_envs[prnt_nd[1]][prnt_nd[2]])])
            indexList = Vector{Vector{Int}}([copy(indices[prnt_nd]), copy(indices[(-prnt_nd[1], prnt_nd[2])]), copy(indices[("top", prnt_nd[1], prnt_nd[2])])])

            for chd_idx in vcat(1:idx_chd-1..., idx_chd+1:TTNKit.number_of_child_nodes(net, prnt_nd)...)
                append!(tensorList, [copy(bottom_envs[prnt_nd[1]][prnt_nd[2]][chd_idx])])
                append!(indexList, [copy(indices[("bottom", prnt_nd[1], prnt_nd[2], chd_idx)])])
            end

            # contract tensors and save the indices of the new top environment
            newIndices, top_envs[ll][pp] = contract_tensors(tensorList, indexList)
            indices[("top", ll, pp)] = newIndices
        end
    end 

    return indices, top_envs
end

function ProjTensorProductOperator(ttn::TreeTensorNetwork{D}, tpo::AbstractTensorProductOperator) where {D}
    net = TTNKit.network(ttn)

    indices = build_Dict(net)
    indices, bottom_envs = bottom_env(ttn, indices, tpo)
    indices, top_envs = top_env(ttn, indices, bottom_envs)

    return ProjTensorProductOperator(net, tpo, bottom_envs, top_envs, indices)
end


function environment(ptpo::ProjTensorProductOperator, pos::Tuple{Int, Int})
    net = ptpo.net
    indices = ptpo.Indices

    dim = dims(codomain(ptpo.tpo.data[1]))[end]

    tensorList = Vector{AbstractTensorMap}([copy(ptpo.top_envs[pos[1]][pos[2]])])
    indexList = Vector{Vector{Int}}([copy(indices[("top", pos[1], pos[2])])])

    append!(tensorList, [Tensor(vcat(zeros(dim-1), 1), (ℂ^dim)'),  Tensor(vcat(1, zeros(dim-1)), ℂ^dim)])
    append!(indexList, [copy(indices[(0, 0)]), copy(indices[(0, 1)])])

    append!(tensorList, [Tensor([1], ℂ^1), Tensor([1], (ℂ^1)')])
    append!(indexList, [copy(indices[(TTNKit.number_of_layers(net)+1, 1)]), copy(indices[(-TTNKit.number_of_layers(net)-1, 1)])])

    for chd_idx in 1:TTNKit.number_of_child_nodes(net, pos)
        append!(tensorList, [copy(ptpo.bottom_envs[pos[1]][pos[2]][chd_idx])])
        append!(indexList, [copy(indices[("bottom", pos[1], pos[2], chd_idx)])])
    end

    newIndices, environment = contract_tensors(tensorList, indexList)

    # permuting resulting environment for easier handling later on
    len = length(newIndices)
    permutation_order = [1:len...]
    sorting_array = []
    for i in 1:len
        append!(sorting_array, [[newIndices[i], permutation_order[i]]])
    end
    sort!(sorting_array, by=first)
    permutation_order = [x[2] for x in sorting_array]

    return TensorKit.permute(environment, Tuple(permutation_order[Int(len/2+1):len]), Tuple(permutation_order[1:Int(len/2)]))
end


function update_environment(ptpo::ProjTensorProductOperator, t::AbstractTensorMap, pos_initial::Tuple{Int, Int}, pos_final::Tuple{Int, Int})
    #check if t has correct index structure
    net = ptpo.net
    indices = ptpo.Indices

    Δ = pos_final .- pos_initial

    if Δ[1] == 1

        tensorList = Vector{AbstractTensorMap}([t, adjoint(t)])
        indexList = Vector{Vector{Int}}([copy(indices[pos_initial]), copy(indices[(-pos_initial[1], pos_initial[2])])])

        for chd in 1:TTNKit.number_of_child_nodes(net, pos_initial)
            append!(tensorList, [copy(ptpo.bottom_envs[pos_initial[1]][pos_initial[2]][chd])])
            append!(indexList, [copy(indices[("bottom", pos_initial[1], pos_initial[2], chd)])])
        end

        newIndices, ptpo.bottom_envs[pos_final[1]][pos_final[2]][TTNKit.index_of_child(net, pos_initial)] = contract_tensors(tensorList, indexList)

    elseif Δ[1] == -1

        idx_chd = TTNKit.index_of_child(net, pos_final)

        tensorList = Vector{AbstractTensorMap}([t, adjoint(t), copy(ptpo.top_envs[pos_initial[1]][pos_initial[2]])])
        indexList = Vector{Vector{Int}}([copy(indices[pos_initial]), copy(indices[(-pos_initial[1], pos_initial[2])]), copy(indices[("top", pos_initial[1], pos_initial[2])])])

        for chd_idx in vcat(1:idx_chd-1..., idx_chd+1:TTNKit.number_of_child_nodes(net, pos_initial)...)
            append!(tensorList, [copy(ptpo.bottom_envs[pos_initial[1]][pos_initial[2]][chd_idx])])
            append!(indexList, [copy(indices[("bottom", pos_initial[1], pos_initial[2], chd_idx)])])
        end

        newIndices, ptpo.top_envs[pos_final[1]][pos_final[2]] = contract_tensors(tensorList, indexList)

    else
        error("Invalid step for updating environments: ", pos_initial, ", ", pos_final)
    end
end