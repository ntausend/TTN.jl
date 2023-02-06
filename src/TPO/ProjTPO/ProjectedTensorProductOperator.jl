struct ProjTPO{N<:TTNKit.AbstractNetwork, T, B<:AbstractBackend} <: AbstractProjTPO{N,T,B}
    net::N
    tpo::TPO
    ortho_center::Vector{Int64} # tracking the current ortho center

    rg_terms_upwards::Vector{Vector{Vector{Vector{Op}}}}
    #rg_id_upwars::Vector{Vector{Vector{T}}}
    environments::Vector{Vector{Vector{Prod{Op}}}}
end

# general constructor from a formal sum, depending on the case it will 
# generate the itensor or tensormap version of this pTPO
function ProjTPO(ttn::TreeTensorNetwork{N, T}, tpo::TPO) where {N,T}
    net = network(ttn)
    # first construct the tpo, invovles some conversion and returns
    # a vector (representing the sum) of the product operators.
    #tpo = _ampo_to_tpo(ampo, physical_lattice(net))
    # calcualte the uprg flows of the operators and identities
    rg_flw_up, id_up_rg = _up_rg_flow(ttn, tpo)
    # build the environments
    envs = _build_environments(ttn, rg_flw_up, id_up_rg)
    return ProjTPO{N, T, backend(ttn)}(net, tpo, vcat(ortho_center(ttn)...), rg_flw_up, envs)
end

environments(ptpo::ProjTPO, pos::Tuple{Int, Int}) = ptpo.environments[pos[1]][pos[2]]

include("./general_utils.jl")
include("./construct_environments_itensors.jl")


function rebuild_environments!(projTPO::ProjTPO, ttn::TreeTensorNetwork)
    net = network(projTPO)
    @assert net == network(ttn)

    tpo = projTPO.tpo
    rg_flw_up, id_up_rg = _up_rg_flow(ttn, tpo)
    # build the environments
    envs = _build_environments(ttn, rg_flw_up, id_up_rg)

    projTPO.rg_terms_upwards .= rg_flw_up
    projTPO.environments     .= envs
    projTPO.ortho_center     .= ortho_center(ttn)

    return projTPO
end

function update_environments!(projTPO::ProjTPO{N, ITensor}, isom::ITensor, pos::Tuple{Int, Int}, pos_final::Tuple{Int, Int}) where{N}

    # pos_final has to be either a child node or the parent node of pos
    @assert pos_final ∈ vcat(child_nodes(network(projTPO), pos), parent_node(network(projTPO), pos))
    
    
    # get the envrionments of the current position
    envs_cur = projTPO[pos]
    envs_target = projTPO[pos_final]
    # extract all components comming from below
    terms_filtered = map(envs_cur) do smt
        filter(T -> site(T) != pos_final, terms(smt))
    end

    # do the rg flow of the terms w.r.t the isometry
    rg_flw = map(terms_filtered) do smt
      _ops = which_op.(smt)
        op_reduction = length(filter(p -> !p, getindex.(params.(smt), :is_identity)))
        if op_reduction > 0
            # if op_reduction == 0 we only have a flow of identities
            op_reduction -= 1
        end
        tensor_list = [isom, _ops..., dag(prime(isom))]
        opt_seq = optimal_contraction_sequence(tensor_list)
				
        _rg_op = contract(tensor_list; sequence = opt_seq)
        prm   = params.(smt)
		# summand index, should be the same for all
		sid  = only(unique(getindex.(prm, :sm)))
        op_length = only(unique(getindex.(prm, :op_length)))
        is_identity = all(getindex.(prm, :is_identity))
		Op(_rg_op, pos; sm = sid, op_length = op_length-op_reduction, is_identity = is_identity)
    end
    # collapse the onsite operators
    rg_flw = _collapse_onsite(rg_flw)

    # now rebuild the environments for the new mode
    # first split rg_flw terms according to identity, onsite and interaction term
    rg_flw_id     = only(filter(T -> getindex(params(T), :is_identity), rg_flw))
    rg_flw_nonid  = filter(T -> !getindex(params(T), :is_identity), rg_flw)
    rg_flw_onsite = only(filter(T -> isone(getindex(params(T), :op_length)), rg_flw_nonid))
    rg_flw_int    = filter(T -> !isone(getindex(params(T), :op_length)), rg_flw_nonid)
    
    # identity needs to be attached to the terms from below
    env_n_id = filter(envs_target) do trm
        trm_smid = only(unique(getindex.(params.(terms(trm)),:sm)))
        all(s -> s ∈ getindex(params(rg_flw_id), :sm), trm_smid)
    end
    # now replace the identity with the updated one
    env_n_id = map(env_n_id) do trm
        #trm_t = terms(trm)
        trm_t = terms(trm)
        trm_smid  = only(unique(getindex.(params.(terms(trm)),:sm)))
        op_length = only(unique(getindex.(params.(terms(trm)),:op_length)))
        id_n  = Op(which_op(rg_flw_id), site(rg_flw_id); sm = trm_smid, is_identity = true, op_length = op_length)
        reduce(*, vcat(filter(T -> site(T) != site(rg_flw_id), trm_t), id_n), init = Prod{Op}())
    end

    # the onsite operator needs to be padded with identities from below
    env_onsite_old = only(filter(terms.(envs_target)) do trm
        trm_smid = only(unique(getindex.(params.(trm),:sm)))
        trm_smid == getindex(params(rg_flw_onsite),:sm)
    end)
    env_n_onsite = reduce(*, vcat(filter(T -> site(T) != site(rg_flw_onsite), env_onsite_old), rg_flw_onsite), init = Prod{Op}())
    
    # rest of the interaction terms are handled similar
    env_n_int = map(rg_flw_int) do rg_trm
        trms_old = only(filter(terms.(envs_target)) do trm
            trm_smid = only(unique(getindex.(params.(trm),:sm)))
            trm_smid == getindex(params(rg_trm), :sm)
        end)
        reduce(*, vcat(filter(T -> site(T) != site(rg_trm),trms_old), rg_trm), init = Prod{Op}())
    end

    # now rebuild the environments for the new node
    projTPO.environments[pos_final[1]][pos_final[2]] = vcat(env_n_onsite, env_n_id..., env_n_int)

    return projTPO
end


function ∂A(projTPO::ProjTPO{N, ITensor}, pos::Tuple{Int,Int}) where{N}
    # getting the enviornments of the current position
    envs = projTPO[pos]

    function action(T::ITensor)
        mapreduce(+, envs) do trm
            _ops = which_op.(trm)
            tensor_list = vcat(T, _ops)
            opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
            return noprime(contract(tensor_list; sequence = opt_seq))
        end
    end
end


function ∂A2(projTPO::ProjTPO{N, ITensor}, isom::ITensor, posi::Tuple{Int,Int}) where N
    envs = projTPO[posi]
    function action(link::ITensor)
        mapreduce(+, envs) do trm 
            tensor_list = vcat(isom, dag(prime(isom)), link, which_op.(trm))
            opt_seq = optimal_contraction_sequence(tensor_list)
            return noprime(contract(tensor_list; sequence = opt_seq))
        end
    end
    return action
end
