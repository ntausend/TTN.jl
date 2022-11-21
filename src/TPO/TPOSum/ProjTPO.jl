using TTNKit: number_of_child_nodes, number_of_sites, number_of_layers, number_of_tensors, index_of_child, eachindex, eachlayer, parent_node, child_nodes

struct ProjTensorProductOperator
    net::AbstractNetwork
    tpo::AbstractTensorProductOperator

    bottomEnvironment::Vector{Vector{Vector{AbstractTensorMap}}}
    topEnvironment::Vector{Vector{AbstractTensorMap}}
 
    bottomIndices::Vector{Vector{Vector{Vector{Int64}}}}
    topIndices::Vector{Vector{Vector{Int64}}}
end

function Base.copy(ptpo::ProjTensorProductOperator)
    netc = deepcopy(ptpo.net)
    tpoc = deepcopy(ptpo.tpo)
    bottomEnvironmentc = deepcopy(ptpo.bottomEnvironment)
    topEnvironmentc = deepcopy(ptpo.topEnvironment)
    bottomIndicesc = deepcopy(ptpo.bottomIndices)
    topIndicesc = deepcopy(ptpo.topIndices)
    return ProjTensorProductOperator(netc, tpoc, bottomEnvironmentc, topEnvironmentc, bottomIndicesc, topIndicesc)
end

function contract_tensors(tensorList::Vector{AbstractTensorMap}, indexList::Vector{Vector{Int}})
    uniqueIndices = Any[]
    doubleIndices = Any[]
    flatIndexList = vcat(indexList...)  

    while !(isempty(flatIndexList))
        el = popfirst!(flatIndexList)
        if !(el in flatIndexList) && !(el in doubleIndices)
            append!(uniqueIndices, [el])
        else
            append!(doubleIndices, [el])
        end
    end

    contractList = map(indexList) do list
        map(list) do pp
            return pp in uniqueIndices ? -findall(isequal(pp), uniqueIndices)[1] : findall(isequal(pp), doubleIndices)[1]
        end
    end

    return (uniqueIndices, @ncon(tensorList, contractList))
end

num_chds(net::AbstractNetwork, pos::Tuple{Int,Int}) = number_of_child_nodes(net, pos)
int_index(net::AbstractNetwork, pos::Tuple{Int,Int}) = internal_index_of_legs(net, pos)

function bottomEnvironment(ttn::TreeTensorNetwork, tpo::AbstractTensorProductOperator)
    
    net = TTNKit.network(ttn)

    n_sites = number_of_sites(net)
    n_tensors = number_of_tensors(net) + n_sites

    # first two vectors are for layer and position within the layer respectivley, 
    # third one is for enumerating the bottom environments - one for each child leg

    bottomEnvironment = Vector{Vector{Vector{AbstractTensorMap}}}(undef, number_of_layers(net)) 
    bottomIndices = Vector{Vector{Vector{Vector{Int}}}}(undef, number_of_layers(net)) 

    # first layer
    bottomEnvironment[1] = Vector{Vector{AbstractTensorMap}}([[tpo.data[n] for n in 1:num_chds(net, (1,pp))] for pp in eachindex(net,1)])
    bottomIndices[1] = Vector{Vector{Vector{Int}}}([[[int_index(net, (1,pp))[n], int_index(net, (1,pp))[n]+n_tensors, -(int_index(net, (1,pp))[n]), -(int_index(net, (1,pp))[n]+1)] for n in 1:num_chds(net,(1,pp))] for pp in eachindex(net,1)])

    # contract tensors at both ends of the MPO chain
    dim = dims(codomain(tpo.data[1]))[end]
    (bottomIndices[1][1][1], bottomEnvironment[1][1][1]) = contract_tensors(Vector{AbstractTensorMap}([bottomEnvironment[1][1][1], Tensor(vcat(zeros(dim-1), 1), (ℂ^dim)')]), [bottomIndices[1][1][1],[-1]])
    (bottomIndices[1][end][end], bottomEnvironment[1][end][end]) = contract_tensors(Vector{AbstractTensorMap}([bottomEnvironment[1][end][end], Tensor(vcat(1, zeros(dim-1)), ℂ^dim)]), [bottomIndices[1][end][end],[-number_of_sites(net)-1]])

    for ll in Iterators.drop(eachlayer(net), 1)
        bottomEnvironment[ll] = Vector{Vector{AbstractTensorMap}}(undef, number_of_tensors(net, ll))
        bottomIndices[ll] = Vector{Vector{Vector{Int}}}(undef, number_of_tensors(net, ll))

        for pp in eachindex(net, ll)
            n_chds = num_chds(net, (ll,pp))
            bottomEnvironment[ll][pp] = Vector{AbstractTensorMap}(undef, n_chds)
            bottomIndices[ll][pp] = Vector{Vector{Int}}(undef, n_chds)

            for chd_nd in child_nodes(net, (ll,pp))
                chd_idx = index_of_child(net, chd_nd)

                tensorListTTN = Vector{AbstractTensorMap}([copy(ttn[(chd_nd)]), adjoint(copy(ttn[(chd_nd)]))])
                tensorListBottom = Vector{AbstractTensorMap}([copy(bottomEnvironment[chd_nd[1]][chd_nd[2]][index_of_child(net, chd_chd_nd)]) for chd_chd_nd in child_nodes(net, chd_nd)])
                indexListTTN = Vector{Vector{Int}}([int_index(net, chd_nd), int_index(net, chd_nd)[vcat(end,1:end-1)].+n_tensors])
                indexListBottom = Vector{Vector{Int}}([copy(bottomIndices[chd_nd[1]][chd_nd[2]][index_of_child(net, chd_chd_nd)]) for chd_chd_nd in child_nodes(net, chd_nd)])

                (bottomIndices[ll][pp][chd_idx], bottomEnvironment[ll][pp][chd_idx]) = contract_tensors(vcat(tensorListTTN, tensorListBottom), vcat(indexListTTN, indexListBottom))
            end        
        end
    end 

    return bottomIndices, bottomEnvironment
end

function topEnvironment(ttn::TreeTensorNetwork{D}, bottomEnvironment::Vector{Vector{Vector{AbstractTensorMap}}}, bottomIndices::Vector{Vector{Vector{Vector{Int64}}}}) where {D}
    
    net = TTNKit.network(ttn)
    n_sites = number_of_sites(net)
    n_tensors = number_of_tensors(net) + n_sites

    # first two vectors are for layer and position within the layer respectivley
    topEnvironment = Vector{Vector{AbstractTensorMap}}(undef, number_of_layers(net))

    topIndices = Vector{Vector{Vector{Int}}}(undef, number_of_layers(net)) 
    topIndices[number_of_layers(net)] = Vector{Vector{Int}}(undef, number_of_tensors(net, 1))

    # top environment of the top node
    topEnvironment[number_of_layers(net)] = [Tensor([1], ℂ^1*(ℂ^1)')]
    topIndices[number_of_layers(net)][1] = [n_tensors, 2*n_tensors] 

    for ll in Iterators.drop(reverse(eachlayer(net)), 1)
        topEnvironment[ll] = Vector{AbstractTensorMap}(undef, number_of_tensors(net, ll))
        topIndices[ll] = Vector{Vector{Int}}(undef, number_of_tensors(net, ll))

        for pp in eachindex(net, ll)
            prnt_nd = parent_node(net, (ll,pp))
            idx_chd = index_of_child(net, (ll,pp))

            # top environment of node (ll,pp) is built by contracting the parent node with its top environment and its remaining bottom environments
            tensorListTTN = Vector{AbstractTensorMap}([copy(ttn[(prnt_nd)]), copy(adjoint(ttn[(prnt_nd)])), copy(topEnvironment[prnt_nd[1]][prnt_nd[2]])])
          tensorListBottom = Vector{AbstractTensorMap}([copy(bottomEnvironment[prnt_nd[1]][prnt_nd[2]][chd_nd]) for chd_nd in deleteat!(collect(1:num_chds(net, prnt_nd)), idx_chd)])
            indexListTTN = Vector{Vector{Int}}([int_index(net, prnt_nd), int_index(net, prnt_nd)[vcat(end,1:end-1)].+n_tensors, topIndices[prnt_nd[1]][prnt_nd[2]]])
            indexListBottom = Vector{Vector{Int}}([copy(bottomIndices[prnt_nd[1]][prnt_nd[2]][chd_nd]) for chd_nd in deleteat!(collect(1:num_chds(net, prnt_nd)), idx_chd)])

            # contract tensors and save the indices of the new top environment
            (topIndices[ll][pp], topEnvironment[ll][pp]) = contract_tensors(vcat(tensorListTTN, tensorListBottom), vcat(indexListTTN, indexListBottom))
        end
    end 

    return topIndices, topEnvironment
end

function ProjTensorProductOperator(ttn::TreeTensorNetwork{D}, tpo::AbstractTensorProductOperator) where {D}
    net = TTNKit.network(ttn)

    bottomIndices, bottomEnvironmentc = bottomEnvironment(ttn, tpo)
    topIndices, topEnvironmentc = topEnvironment(ttn, bottomEnvironmentc, bottomIndices)

    return ProjTensorProductOperator(net, tpo, bottomEnvironmentc, topEnvironmentc, bottomIndices, topIndices)
end


function environment(ptpo::ProjTensorProductOperator, pos::Tuple{Int, Int})
    net = ptpo.net
    topIndices = ptpo.topIndices
    bottomIndices = ptpo.bottomIndices

    dim = dims(codomain(ptpo.tpo.data[1]))[end]

    tensorList = Vector{AbstractTensorMap}(vcat([copy(ptpo.topEnvironment[pos[1]][pos[2]])], [copy(ptpo.bottomEnvironment[pos[1]][pos[2]][chd_nd]) for chd_nd in 1:num_chds(net,pos)]))
    indexList = Vector{Vector{Int}}(vcat([copy(topIndices[pos[1]][pos[2]])], [copy(bottomIndices[pos[1]][pos[2]][chd_nd]) for chd_nd in 1:num_chds(net,pos)]))

    (newIndices, environment) = contract_tensors(tensorList, indexList)

    # permuting indices of the resulting environment for easier handling later on
    len = length(newIndices)
    permutationOrder = [1:len...]
    sortingArray = map((x,y) -> [x,y], newIndices, permutationOrder)
    sort!(sortingArray, by=first)
    permutationOrder = [s[2] for s in sortingArray]

    return TensorKit.permute(environment, Tuple(permutationOrder[Int(len/2+1):len]), Tuple(permutationOrder[1:Int(len/2)]))
end


function update_environment(ptpo::ProjTensorProductOperator, t::AbstractTensorMap, pos_initial::Tuple{Int, Int}, pos_final::Tuple{Int, Int})
    #check if t has correct index structure
    net = ptpo.net
    n_sites = number_of_sites(net)
    n_tensors = number_of_tensors(net) + n_sites

    topIndices = ptpo.topIndices
    bottomIndices = ptpo.bottomIndices

    Δ = pos_final .- pos_initial

    if Δ[1] == 1

        tensorListTTN = Vector{AbstractTensorMap}([t, adjoint(t)])
        indexListTTN = Vector{Vector{Int}}([int_index(net, pos_initial), int_index(net, pos_initial)[vcat(end,1:end-1)].+n_tensors])
        tensorListBottom = Vector{AbstractTensorMap}([copy(ptpo.bottomEnvironment[pos_initial[1]][pos_initial[2]][chd_nd]) for chd_nd in 1:num_chds(net,pos_initial)])
        indexListBottom = Vector{Vector{Int}}([copy(bottomIndices[pos_initial[1]][pos_initial[2]][chd_nd]) for chd_nd in 1:num_chds(net,pos_initial)])

        nothing, ptpo.bottomEnvironment[pos_final[1]][pos_final[2]][index_of_child(net, pos_initial)] = contract_tensors(vcat(tensorListTTN, tensorListBottom), vcat(indexListTTN, indexListBottom))

    elseif Δ[1] == -1

        idx_chd = index_of_child(net, pos_final)

        tensorListTTN = Vector{AbstractTensorMap}([t, adjoint(t), copy(ptpo.topEnvironment[pos_initial[1]][pos_initial[2]])])
        indexListTTN = Vector{Vector{Int}}([int_index(net, pos_initial), int_index(net, pos_initial)[vcat(end,1:end-1)].+n_tensors, copy(topIndices[pos_initial[1]][pos_initial[2]])])
        tensorListBottom = Vector{AbstractTensorMap}([copy(ptpo.bottomEnvironment[pos_initial[1]][pos_initial[2]][chd_nd]) for chd_nd in deleteat!(collect(1:num_chds(net, pos_initial)), idx_chd)])
        indexListBottom = Vector{Vector{Int}}([copy(bottomIndices[pos_initial[1]][pos_initial[2]][chd_nd]) for chd_nd in deleteat!(collect(1:num_chds(net, pos_initial)), idx_chd)])

        (nothing, ptpo.topEnvironment[pos_final[1]][pos_final[2]]) = contract_tensors(vcat(tensorListTTN, tensorListBottom), vcat(indexListTTN, indexListBottom))

    else
        error("Invalid step for updating environments: ", pos_initial, ", ", pos_final)
    end
end
