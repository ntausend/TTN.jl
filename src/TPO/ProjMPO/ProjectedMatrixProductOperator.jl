# make this abstract and derive from that the ProjMatrixProductOperator
struct ProjMPO{N<:AbstractNetwork, T, B<:AbstractBackend} <: AbstractProjTPO{N,T,B}
    net::N
    tpo::AbstractTensorProductOperator
    ortho_center :: Vector{Int}

    bottom_envs::Vector{Vector{Vector{T}}}
    top_envs::Vector{Vector{T}}

    bottom_indices::Vector{Vector{Vector{Vector{Int64}}}}
    top_indices::Vector{Vector{Vector{Int64}}}
end

bottom_environment(projTPO::ProjMPO, pos::Tuple{Int,Int}) = projTPO.bottom_envs[pos[1]][pos[2]]
bottom_environment(projTPO::ProjMPO, pos::Tuple{Int,Int}, n_child::Int) = bottom_environment(projTPO, pos)[n_child]
top_environment(   projTPO::ProjMPO, pos::Tuple{Int,Int}) = projTPO.top_envs[pos[1]][pos[2]]

bottom_indices(projTPO::ProjMPO, pos::Tuple{Int, Int}) = projTPO.bottom_indices[pos[1]][pos[2]]
bottom_indices(projTPO::ProjMPO, pos::Tuple{Int, Int}, n_child::Int) = bottom_indices(projTPO,pos)[n_child]
top_indices(   projTPO::ProjMPO, pos::Tuple{Int, Int}) = projTPO.top_indices[pos[1]][pos[2]]

# compact return of the environments of one tuple
environments(projTPO::ProjMPO, pos::Tuple{Int, Int}) = vcat(bottom_environment(projTPO, pos)..., top_environment(projTPO, pos))
# indices on that spot
indices(projTPO::ProjMPO, pos::Tuple{Int, Int}) = vcat(bottom_indices(projTPO, pos)..., top_indices(projTPO, pos))
 


include("./constructing_projmpo_from_mpo_tensorkit.jl")
include("./constructing_projmpo_from_mpo_itensors.jl")


function ProjMPO(ttn::TreeTensorNetwork{N, T}, tpo::AbstractTensorProductOperator) where{N, T}
    # sanity check if the physical setup is correct
    @assert physical_lattice(network(ttn)) == lattice(tpo)

    bInd, bEnv = _construct_bottom_environments(ttn, tpo)
    tInd, tEnv = _construct_top_environments(ttn, bEnv, bInd)

    return ProjMPO{N, T, backend(tpo)}(network(ttn), tpo, vcat(ortho_center(ttn)...), bEnv, tEnv, bInd, tInd)
end

function rebuild_environments!(projTPO::ProjMPO, ttn::TreeTensorNetwork)
    net = network(projTPO)
    @assert net == network(ttn)

    tpo = projTPO.tpo
    bInd, bEnv = _construct_bottom_environments(ttn, tpo)
    tInd, tEnv = _construct_top_environments(ttn, bEnv, bInd)

    projTPO.bottom_envs .= bEnv
    projTPO.top_envs    .= tEnv

    projTPO.bottom_indices .= bInd
    projTPO.top_indices    .= tInd
    projTPO.ortho_center   .= ortho_center(ttn)

    return projTPO
end



function update_environments!(projTPO::ProjMPO, isom, pos::Tuple{Int,Int}, pos_final::Tuple{Int, Int})
    dir = pos_final .- pos
    if dir[1] == 1
        return _update_bottom_environment!(projTPO, isom, pos, pos_final)
    else
        @assert dir[1] == -1
        return _update_top_environment!(projTPO, isom, pos, pos_final)
    end
end

# how to do this abstractly for arbitrary networks??
function _update_top_environment!(projTPO::ProjMPO{N, TensorMap}, isom::TensorMap, pos::Tuple{Int,Int}, pos_final::Tuple{Int, Int}) where{N}
    net = network(projTPO)
    n_sites = number_of_sites(net)
    n_tensors = number_of_tensors(net) + n_sites

    tensorListTTN = [isom, adjoint(isom),top_environment(projTPO, pos)]
    int_leg = internal_index_of_legs(net, pos)
    tInd = [int_leg, vcat(int_leg[end], int_leg[1:end-1]) .+ n_tensors, top_indices(projTPO, pos)]
    b_collect = deleteat!(collect(1:number_of_child_nodes(net, pos)), index_of_child(net, pos_final))
    tensorListBottom = map(jj -> bottom_environment(projTPO, pos, jj), b_collect)
    bInd = map(jj -> bottom_indices(projTPO, pos, jj), b_collect)

    new_inds, projTPO.top_envs[pos_final[1]][pos_final[2]] = contract_tensors(vcat(tensorListTTN, tensorListBottom), vcat(tInd, bInd))
    
    @assert new_inds == top_indices(projTPO, pos_final)
    return projTPO
end
function _update_bottom_environment!(projTPO::ProjMPO{N, TensorMap}, isom::TensorMap, pos::Tuple{Int,Int}, pos_final::Tuple{Int,Int}) where {N}
    net = network(projTPO)
    n_sites = number_of_sites(net)
    n_tensors = number_of_tensors(net) + n_sites

    int_leg = internal_index_of_legs(net, pos)
    tInd    = [int_leg, vcat(int_leg[end], int_leg[1:end-1]) .+ n_tensors]
    tensorListTTN = [isom, adjoint(isom)]
    bInd = bottom_indices(projTPO, pos)
    tensorListBottom = bottom_environment(projTPO, pos)

    new_inds, projTPO.bottom_envs[pos_final[1]][pos_final[2]][index_of_child(net, pos)] = contract_tensors(vcat(tensorListTTN, tensorListBottom), vcat(tInd, bInd))
    @assert new_inds == bottom_indices(projTPO, pos_final, index_of_child(net, pos))
    return projTPO
end

function _update_top_environment!(projTPO::ProjMPO{N, ITensor}, isom::ITensor, pos::Tuple{Int,Int}, pos_final::Tuple{Int, Int}) where{N}
    net = network(projTPO)
    tEnv = top_environment(projTPO, pos)
    b_collect = deleteat!(collect(1:number_of_child_nodes(net, pos)), index_of_child(net, pos_final))
    tensorListBottom = map(jj -> bottom_environment(projTPO, pos, jj), b_collect)

    opt_seq = ITensors.optimal_contraction_sequence(isom, dag(prime(isom)), tEnv, tensorListBottom...)
    projTPO.top_envs[pos_final[1]][pos_final[2]] = contract(isom, dag(prime(isom)), tEnv, tensorListBottom...; sequence = opt_seq)

    return projTPO
end
function _update_bottom_environment!(projTPO::ProjMPO{N, ITensor}, isom::ITensor, pos::Tuple{Int,Int}, pos_final::Tuple{Int,Int}) where {N}
    net = network(projTPO)

    bEnvs = bottom_environment(projTPO, pos)
    opt_seq = ITensors.optimal_contraction_sequence(isom, dag(prime(isom)), bEnvs...)
    projTPO.bottom_envs[pos_final[1]][pos_final[2]][index_of_child(net, pos)] = contract(isom, dag(prime(isom)), bEnvs...; sequence = opt_seq)
    return projTPO
end


# action of the TPO on the removed A tensor at position p

function ∂A(projTPO::ProjMPO{N, TensorMap}, pos::Tuple{Int,Int}) where{N}
    tEnv  = top_environment(projTPO, pos)
    bEnvs = bottom_environment(projTPO, pos)
    ttn_coord = internal_index_of_legs(network(projTPO), pos)
    tInds = top_indices(projTPO, pos)
    bInds = bottom_indices(projTPO, pos)
    function action(T::AbstractTensorMap)
        inds, res = contract_tensors(vcat(T, bEnvs), vcat([ttn_coord], bInds))
        inds, res = contract_tensors([res, tEnv], [inds, tInds])

        # now permute back, top node should have the largest value and childs
        # should be ordered
        perm = collect(1:length(inds))[sortperm(inds)]
        return TensorKit.permute(res, Tuple(perm[1:end-1]), (perm[end],))
    end
    return action
end


function ∂A(projTPO::ProjMPO{N, ITensor}, pos::Tuple{Int,Int}) where{N}
    envs = projTPO[pos]
    function action(T::ITensor)
        tensor_list = vcat(T, envs)
        opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
        return noprime(contract(tensor_list; sequence = opt_seq))
    end
    return action
end


# action of the TPO on the removed link tensor between positions posi (initial) and posf (final)
function ∂A2(projTPO::ProjMPO{N, TensorMap}, isom::TensorMap, posi::Tuple{Int,Int}, posf::Tuple{Int,Int}) where N
    net = network(projTPO)
    tEnv = top_environment(projTPO, posi)
    bEnvs = bottom_environment(projTPO, posi)
    ttn_coord = Vector{Float64}(internal_index_of_legs(net, posi))
    tInds = top_indices(projTPO, posi)
    bInds = bottom_indices(projTPO, posi)

    if posf ∈ child_nodes(projTPO.net, posi)
        idx = index_of_child(projTPO.net, posf)
        r_coord = [ttn_coord[idx], ttn_coord[idx]+0.5]
        ttn_coord[idx] += 0.5 
    else 
        idx = 1+number_of_child_nodes(projTPO.net, posi)
        r_coord = [ttn_coord[idx]-0.5, ttn_coord[idx]]
        ttn_coord[idx] -= 0.5 
    end
    n_tensors = number_of_tensors(projTPO.net) + number_of_sites(projTPO.net)
    function action(link::TensorMap)
        if posf ∈ child_nodes(projTPO.net, posi)
            idx = index_of_child(projTPO.net, posf)
            n_chds = number_of_child_nodes(net, posi)

            bEnvsSplit = map(chd_nd -> bEnvs[chd_nd], deleteat!(collect(1:n_chds), idx))
            bIndsSplit = map(chd_nd -> bInds[chd_nd], deleteat!(collect(1:n_chds), idx))

            inds, res = contract_tensors([isom, tEnv], [ttn_coord, tInds])
            inds, res = contract_tensors(vcat(res, bEnvsSplit),vcat([inds], bIndsSplit))
            inds, res = contract_tensors([res, link], [inds, r_coord])
            inds, res = contract_tensors([res, bEnvs[idx]], [inds, bInds[idx]])
            inds, res = contract_tensors([res, adjoint(isom)], [inds, ttn_coord[vcat(end,1:end-1)].+n_tensors])
        else    
            inds, res = contract_tensors([isom, link], [ttn_coord, r_coord])
            inds, res = contract_tensors([res, tEnv], [inds, tInds])
            inds, res = contract_tensors(vcat(res, bEnvs), vcat([inds], bInds))
            inds, res = contract_tensors([res, adjoint(isom)], [inds, ttn_coord[vcat(end,1:end-1)].+n_tensors])
        end

        perm = collect(1:length(inds))[sortperm(inds)]
        return TensorKit.permute(res, Tuple(perm[1:end-1]), (perm[end],))
    end
    return action
end

function ∂A2(projTPO::ProjMPO{N, ITensor}, isom::ITensor, posi::Tuple{Int,Int}) where N
    envs = projTPO[posi]
    function action(link::ITensor)
        tensor_list = vcat(isom, dag(prime(isom)), link, envs...)
        opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
        return noprime(contract(tensor_list; sequence = opt_seq))
    end
    return action
end
