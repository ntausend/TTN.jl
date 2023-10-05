# make this abstract and derive from that the ProjMatrixProductOperator
struct ProjMPO{N<:AbstractNetwork, T} <: AbstractProjTPO{N,T}
    net::N
    tpo::MPOWrapper
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


#include("./constructing_projmpo_from_mpo_tensorkit.jl")
include("./constructing_projmpo_from_mpo_itensors.jl")


function ProjMPO(ttn::TreeTensorNetwork{N, T}, tpo::MPOWrapper) where{N, T}
    # sanity check if the physical setup is correct
    @assert physical_lattice(network(ttn)) == lattice(tpo)

    bInd, bEnv = _construct_bottom_environments(ttn, tpo)
    tInd, tEnv = _construct_top_environments(ttn, bEnv, bInd)

    return ProjMPO{N, T}(network(ttn), tpo, vcat(ortho_center(ttn)...), bEnv, tEnv, bInd, tInd)
end
ProjectedTensorProductOperator(ttn::TreeTensorNetwork, tpo::MPOWrapper) = ProjMPO(ttn, tpo)

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

function _update_top_environment!(projTPO::ProjMPO, isom::ITensor, pos::Tuple{Int,Int}, pos_final::Tuple{Int, Int})
    net = network(projTPO)
    tEnv = top_environment(projTPO, pos)
    b_collect = deleteat!(collect(1:number_of_child_nodes(net, pos)), index_of_child(net, pos_final))
    tensorListBottom = map(jj -> bottom_environment(projTPO, pos, jj), b_collect)

    opt_seq = ITensors.optimal_contraction_sequence(isom, dag(prime(isom)), tEnv, tensorListBottom...)
    projTPO.top_envs[pos_final[1]][pos_final[2]] = contract(isom, dag(prime(isom)), tEnv, tensorListBottom...; sequence = opt_seq)

    return projTPO
end
function _update_bottom_environment!(projTPO::ProjMPO, isom::ITensor, pos::Tuple{Int,Int}, pos_final::Tuple{Int,Int})
    net = network(projTPO)

    bEnvs = bottom_environment(projTPO, pos)
    opt_seq = ITensors.optimal_contraction_sequence(isom, dag(prime(isom)), bEnvs...)
    projTPO.bottom_envs[pos_final[1]][pos_final[2]][index_of_child(net, pos)] = contract(isom, dag(prime(isom)), bEnvs...; sequence = opt_seq)
    return projTPO
end

function ∂A(projTPO::ProjMPO, pos::Tuple{Int,Int})
    envs = projTPO[pos]
    function action(T::ITensor)
        tensor_list = vcat(T, envs)
        opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
        return noprime(contract(tensor_list; sequence = opt_seq))
    end
    return action
end


function ∂A2(projTPO::ProjMPO, isom::ITensor, pos::Tuple{Int,Int})
    envs = projTPO[pos]
    function action(link::ITensor)
        tensor_list = vcat(isom, dag(prime(isom)), link, envs...)
        opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
        return noprime(contract(tensor_list; sequence = opt_seq))
    end
    return action
end

function ∂A3(projTPO::ProjMPO, pos::Tuple{Int,Int})
    nextpos = parent_node(projTPO.net, pos)
    bottomEnvs_chd = projTPO.bottom_envs[pos[1]][pos[2]] #bottom_environment(ProjMPO, pos)
    idx = index_of_child(projTPO.net, pos)
    n_chds = number_of_child_nodes(projTPO.net, nextpos)

    envs_prnt = map(nd -> environments(projTPO, nextpos)[nd], deleteat!(collect(1:n_chds+1), idx))

    function action(isom_chd::ITensor, isom_prnt::ITensor)
        tensor_list = vcat(isom_chd, isom_prnt, bottomEnvs_chd..., envs_prnt...)
        opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
        return noprime(contract(tensor_list; sequence = opt_seq))
    end
    return action
end


function noiseterm(ptpo::ProjMPO, T::ITensor, pos_next::Union{Nothing, Tuple{Int, Int}})
    isnothing(pos_next) && return nothing
    pos = ortho_center(ptpo)

    Δpos = pos_next .- pos
    # get direction of the sweep
    dir = Δpos[1]

    # direction is upwards, get all bottom environemnts for the current node
    if dir == 1
        benvs = bottom_environment(ptpo, pos)
        # now contract the envs with the tensor
        tensor_list = vcat(T, benvs)
    elseif dir == -1 && pos_next ∈ child_nodes(network(ptpo), pos)
        # we need to go towards a child
        # collect all environments, except for that one child node
        b_collect = deleteat!(collect(1:number_of_child_nodes(network(ptpo), pos)), index_of_child(network(ptpo), pos_next))
        benvs = map(jj -> bottom_environment(ptpo, pos, jj), b_collect)
        tensor_list = vcat(T, benvs..., top_environment(ptpo, pos))
    else
        error("Next position is not valid for defining a noise term (needs to be neighbored): pos=$pos, next position=$(pos_next)")
    end
    opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
    nt = contract(tensor_list; sequence = opt_seq)

    nt = nt * dag(noprime(nt))
    return nt
end