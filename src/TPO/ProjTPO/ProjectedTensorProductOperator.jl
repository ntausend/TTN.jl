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
    # calcualte the uprg flows of the operators and identities
    rg_flw_up, id_up_rg = _up_rg_flow(ttn, tpo)
    # build the environments
    envs = _build_environments(ttn, rg_flw_up, id_up_rg)
    return ProjTPO{N, T, backend(ttn)}(net, tpo, vcat(ortho_center(ttn)...), rg_flw_up, envs)
end

ProjectedTensorProductOperator(ttn::TreeTensorNetwork, tpo::TPO) = ProjTPO(ttn, tpo)

environments(ptpo::ProjTPO, pos::Tuple{Int, Int}) = ptpo.environments[pos[1]][pos[2]]

include("./general_utils.jl")
include("./construct_environments_itensors.jl")
include("./construct_environments_itensors_implicit_identity.jl")


function rebuild_environments!(projTPO::ProjTPO, ttn::TreeTensorNetwork)
    net = network(projTPO)
    @assert net == network(ttn)

    tpo = projTPO.tpo
    rg_flw_up, id_up_rg = _up_rg_flow(ttn, tpo)
    #rg_flw_up = _up_rg_flow_future(ttn, tpo)
    # build the environments
    envs = _build_environments(ttn, rg_flw_up, id_up_rg)
    #envs = _build_environments_future_future(ttn, rg_flw_up)

    projTPO.rg_terms_upwards .= rg_flw_up
    projTPO.environments     .= envs
    projTPO.ortho_center     .= ortho_center(ttn)

    return projTPO
end


function update_environments_future!(projTPO::ProjTPO{N, ITensor}, isom::ITensor, pos::Tuple{Int, Int}, pos_final::Tuple{Int, Int}) where{N}
    # pos_final has to be either a child node or the parent node of pos
    @assert pos_final ∈ vcat(child_nodes(network(projTPO), pos), parent_node(network(projTPO), pos))
    
    # get the envrionments of the current position
    envs_cur = projTPO[pos]
    envs_target = projTPO[pos_final]
    # extract all components comming from below
    terms_filtered = map(envs_cur) do smt
        filter(T -> site(T) != pos_final, terms(smt))
    end

    idx_trgt = commonind(isom, ttn[pos_final])

    # do the rg flow of the terms w.r.t the isometry
    rg_flw = map(terms_filtered) do smt
      _ops = which_op.(smt)
        op_reduction = length(filter(p -> !p, getindex.(params.(smt), :is_identity)))
        if op_reduction > 0
            # if op_reduction == 0 we only have a flow of identities
            op_reduction -= 1
        end

        # get all non-appearing legs 
		open_links = uniqueinds(Tn, map(_o -> vcat(commoninds(_o, isom), idx_trgt), _ops)...)

        

        #tensor_list = [isom, _ops..., dag(prime(isom))]
        #opt_seq = optimal_contraction_sequence(tensor_list)
				
        #_rg_op = contract(tensor_list; sequence = opt_seq)
        _rg_op = prime!(reduce(*, _ops; init = isom), open_links) * dag(prime(isom))
        prm   = params.(smt)
		# summand index, should be the same for all
		sid  = only(unique(getindex.(prm, :sm)))
        op_length = only(unique(getindex.(prm, :op_length)))
        is_identity = all(getindex.(prm, :is_identity))
		Op(_rg_op, pos; sm = sid, op_length = op_length-op_reduction, is_identity = is_identity)
    end
    # collapse the onsite operators
    rg_flw = _collapse_onsite(rg_flw)
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



function noiseterm(ptpo::ProjTPO{N, ITensor}, T::ITensor, pos_next::Union{Nothing, Tuple{Int, Int}}) where{N}

    isnothing(pos_next) && return nothing
    pos = ortho_center(ptpo)


    # getting the link connecting to the next position
    Δpos = pos_next .- pos
    dir = Δpos[1]

    # upmove
    if dir == 1
        idx_next = only(inds(T; tags = "Link, nl=$(pos[1]), np=$(pos[2])"))
    # downmove
    elseif dir == -1 && pos_next ∈ child_nodes(network(ptpo), pos)
        idx_next = only(inds(T; tags = "Link, nl=$(pos_next[1]), np=$(pos_next[2])"))
    else
        error("Next position is not valid for defining a noise term (needs to be neighbored): pos=$pos, next position=$(pos_next)")
    end

    # get all environments for this site
    envs = environments(ptpo, pos)


    
    # we want to filter the terms according to terms only accting to all other links than pos_next
    # interacting terms and total local terms acting on pos_next.
    # all three classes act individually and has to be summed up "squared"
    # on the last two classes we need to replace the non-trivial operator
    # acting on pos_next with an identity

    envs_trm = terms.(envs)
    # all terms acting with identity on pos_next
    trms_lower = filter(envs_trm) do smt
        sites = site.(smt)
        is_id = getindex.(params.(smt), :is_identity)
        idx_pos = findfirst(isequal(pos_next), sites) 
        return is_id[idx_pos]
    end


    trms_int = filter(envs_trm) do smt
        sites = site.(smt)
        is_id = getindex.(params.(smt), :is_identity)
        idx_pos = findfirst(isequal(pos_next), sites) 
        op_length = only(unique(getindex.(params.(smt), :op_length)))
        return !is_id[idx_pos] && op_length > 1
    end

    # now remove the interaction on the non-trivial site
    trms_int = map(trms_int) do smt
        sites = site.(smt)

        idx_pos = findfirst(isequal(pos_next), sites)

        id_pn  = delta(inds(which_op(smt[idx_pos])))
        rpl_op = Op(id_pn, sites[idx_pos], sm = getindex(params(smt[idx_pos]), :sm), is_identity = true, 
                    op_length = getindex(params(smt[idx_pos]), :op_length))
        smt_n = copy(smt)
        smt_n[idx_pos] = rpl_op
        return smt_n
    end


    # filter the operator acting only onsite on pos_next, if it is present at all
    trm_id = filter(envs_trm) do smt
        sites = site.(smt)
        is_id = getindex.(params.(smt), :is_identity)
        idx_pos = findfirst(isequal(pos_next), sites) 
        op_length = only(unique(getindex.(params.(smt), :op_length)))
        return !is_id[idx_pos] && op_length == 1
    end
    # and replace the the onsite operator with an identity
    trm_id = map(trm_id) do smt
        sites = site.(smt)

        idx_pos = findfirst(isequal(pos_next), sites)

        id_pn  = delta(inds(which_op(smt[idx_pos])))
        rpl_op = Op(id_pn, sites[idx_pos], sm = getindex(params(smt[idx_pos]), :sm), is_identity = true, 
                    op_length = getindex(params(smt[idx_pos]), :op_length))
        smt_n = copy(smt)
        smt_n[idx_pos] = rpl_op
        return smt_n
    end

    trms_act = map([trms_lower, trms_int, trm_id]) do trms
        isempty(trms) && return missing
        trm_tmp = mapreduce(+,trms_lower) do smt
            reduce(*, which_op.(smt), init = T)
        end
        return trm_tmp * dag(prime(noprime(trm_tmp), idx_next))
    end

    nt = reduce(+, skipmissing(trms_act))
    return nt

    # now we want to filter all terms which have the identity on the direction to pos_next




    #@show envs[1]
    trms_filtered = filter(terms.(envs)) do trms
        sites = site.(trms)
        is_id = getindex.(params.(trms), :is_identity)
        idx_pos = findfirst(isequal(pos_next), sites) 
        return is_id[idx_pos]
    end

    # now we need to calcualte the noiseterm for all of them and sum them up

    trms_tmp = map(trms_filtered) do trms
        tmp = reduce(*, which_op.(trms), init = T)
        # now get the link pointing to the next index
        #idx_next = 
        return tmp #* dag(prime(noprime(tmp), idx_next))
    end
    nt = mapreduce(+,Iterators.product(trms_tmp, trms_tmp)) do (trm1, trm2)
        trm1 * dag(prime(noprime(trm2), idx_next))
    end

    id_comp = prime(T, uniqueinds(T, idx_next)) * dag(T)

    return nt + id_comp 
end