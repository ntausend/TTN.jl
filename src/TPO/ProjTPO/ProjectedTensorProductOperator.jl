struct ProjTPO{N<:AbstractNetwork, T} <: AbstractProjTPO{N,T}
    net::N
    tpo::TPO
    ortho_center::Vector{Int64} # tracking the current ortho center

    rg_terms_upwards::Vector{Vector{Vector{Vector{Op}}}}
    #rg_id_upwars::Vector{Vector{Vector{T}}}
    environments::Vector{Vector{Vector{Prod{Op}}}}
    save_to_cpu::Bool
end

# general constructor from a formal sum, depending on the case it will 
# generate the itensor or tensormap version of this pTPO
"""
```julia
    ProjTPO(ttn::TreeTensorNetwork{N, T}, tpo::TPO) where {N,T}
```

Builds the link tensors for applying the local action of the Hamiltonian.
"""
function ProjTPO(ttn::TreeTensorNetwork{N, T}, tpo::TPO; save_to_cpu::Bool = false) where {N,T}
    net = network(ttn)
    # first construct the tpo, invovles some conversion and returns
    # a vector (representing the sum) of the product operators.
    # calcualte the uprg flows of the operators and identities
    rg_flw_up, id_up_rg = _up_rg_flow(ttn, tpo)
    # build the environments
    envs = _build_environments(ttn, rg_flw_up, id_up_rg; save_to_cpu)

    return ProjTPO{N, T}(net, tpo, vcat(ortho_center(ttn)...), rg_flw_up, envs, save_to_cpu)
end

ProjectedTensorProductOperator(ttn::TreeTensorNetwork, tpo::TPO; save_to_cpu::Bool = false) = ProjTPO(ttn, tpo; save_to_cpu)

environments(ptpo::ProjTPO, pos::Tuple{Int, Int}) = ptpo.environments[pos[1]][pos[2]]

include("./general_utils.jl")
include("./construct_environments_itensors.jl")
include("./construct_environments_itensors_implicit_identity.jl")

function rebuild_environments!(projTPO::ProjTPO, ttn::TreeTensorNetwork)
    net = network(projTPO)
    @assert net == network(ttn)

    tpo = projTPO.tpo
    rg_flw_up, id_up_rg = _up_rg_flow(ttn, tpo)
    # build the environments
    envs = _build_environments(ttn, rg_flw_up, id_up_rg; save_to_cpu)

    projTPO.rg_terms_upwards .= rg_flw_up
    projTPO.environments     .= envs
    projTPO.ortho_center     .= ortho_center(ttn)

    return projTPO
end


function update_environments_future!(projTPO::ProjTPO, isom::ITensor, pos::Tuple{Int, Int}, pos_final::Tuple{Int, Int})
    save_to_cpu = projTPO.save_to_cpu
    # pos_final has to be either a child node or the parent node of pos
    @assert pos_final ∈ vcat(child_nodes(network(projTPO), pos), parent_node(network(projTPO), pos))
    
    # get the envrionments of the current position
    envs_cur    = save_to_cpu ? convert_cu(projTPO[pos]) : projTPO[pos]
    envs_target = save_to_cpu ? convert_cu(projTPO[pos_final]) : projTPO[pos_final]

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


function update_environments!(projTPO::ProjTPO, isom::ITensor, pos::Tuple{Int, Int}, pos_final::Tuple{Int, Int})
    save_to_cpu = projTPO.save_to_cpu

    # pos_final has to be either a child node or the parent node of pos
    @assert pos_final ∈ vcat(child_nodes(network(projTPO), pos), parent_node(network(projTPO), pos))
    isom = TTN.convert_cu(isom)
    
    
    # get the envrionments of the current position
    # envs_cur = projTPO[pos]
    # envs_target = projTPO[pos_final]

    envs_cur    = save_to_cpu ? convert_cu(projTPO[pos]) : projTPO[pos]
    envs_target = save_to_cpu ? convert_cu(projTPO[pos_final]) : projTPO[pos_final]

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

    # now rbuild the environments for the new mode
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
    env_n_int = map(enumerate(rg_flw_int)) do (i,rg_trm)
        trms_old = only(filter(terms.(envs_target)) do trm
            trm_smid = only(unique(getindex.(params.(trm),:sm)))
            trm_smid == getindex(params(rg_trm), :sm)
        end)
        reduce(*, vcat(filter(T -> site(T) != site(rg_trm),trms_old), rg_trm), init = Prod{Op}())
    end

    # now rebuild the environments for the new node
    new_envs = vcat(env_n_onsite, env_n_id..., env_n_int)
    projTPO.environments[pos_final[1]][pos_final[2]] = save_to_cpu ? convert_cpu(new_envs) : new_envs

    return projTPO
end

"""
```julia
    ∂A(projTPO::ProjTPO, pos::Tuple{Int,Int})
```

Returns the local action of the hamiltonian projected onto the `pos` node in the network.
"""
function ∂A(projTPO::ProjTPO, pos::Tuple{Int,Int})
    # getting the enviornments of the current position
    envs = projTPO.save_to_cpu ? convert_cu(projTPO[pos]) : projTPO[pos]

    function action(T::ITensor)
        result = mapreduce(+, envs) do trm
            _ops = which_op.(trm)
            tensor_list = vcat(T, _ops)
            opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
            return noprime(contract(tensor_list; sequence = opt_seq))
        end
        return result
    end
end

function ∂A(projTPO::ProjTPO{N, ITensor}, o1s::Vector{ITensor}, weight::Float64, pos::Tuple{Int,Int}) where{N}
    # getting the enviornments of the current position
    envs = projTPO.save_to_cpu ? convert_cu(projTPO[pos]) : projTPO[pos]

    #o1s = [inner(sweep_handler.ttn, ttn_ortho, pos) for ttn_ortho in sweep_handler.ttns_ortho]

    function action(T::ITensor)
        
        result = mapreduce(+, envs) do trm
            _ops = which_op.(trm)
            tensor_list = vcat(T, _ops)
            opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
            return noprime(contract(tensor_list; sequence = opt_seq)) 
        end

        orthos = mapreduce(+,o1s) do o1
            return noprime((o1 * dag(noprime(o1))) * T)
        end

        return result + (weight * orthos)
    end
end

"""
```julia
    ∂A2(projTPO::ProjMPO, pos::Tuple{Int,Int})
```

Returns the local action of the hamiltonian projected onto the link between the tensor at the node `pos` and `isom` which is assumed to be placed at one of the nodes connected to `pos` (NOT CHECKT!)
"""
function ∂A2(projTPO::ProjTPO, isom::ITensor, posi::Tuple{Int,Int})
    envs = projTPO.save_to_cpu ? convert_cu(projTPO[posi]) : projTPO[posi]

    function action(link::ITensor)
        result = mapreduce(+, envs) do trm 
            tensor_list = vcat(isom, dag(prime(isom)), link, which_op.(trm))
            opt_seq = optimal_contraction_sequence(tensor_list)
            return noprime(contract(tensor_list; sequence = opt_seq))
        end
        return result
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

    res_inds = uniqueinds(T, idx_next)
    #build the reduced denisty matrix for that site

    rho = prime(T, res_inds) * dag(T)

    # get all environments for this site
    envs = environments(ptpo, pos)

    envs_trm = ITensors.terms.(envs)

	# all terms acting with identity on pos_next
     trms_lower = filter(envs_trm) do smt
        sites = ITensors.site.(smt)
        is_id = getindex.(ITensors.params.(smt), :is_identity)
        idx_pos = findfirst(isequal(pos_next), sites)
        return is_id[idx_pos]
    end

	Tlower =  mapreduce(+, trms_lower) do smt
        ops = map(ITensors.which_op, filter(smt) do op
            site = ITensors.site(op)
            !(site == pos_next)
        end)
        tensor_list = vcat(T, ops)
        opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
        noprime(contract(tensor_list; sequence = opt_seq))
    end

	 trms_int = filter(envs_trm) do smt
        sites = ITensors.site.(smt)
        is_id = getindex.(ITensors.params.(smt), :is_identity)
        idx_pos = findfirst(isequal(pos_next), sites)
        op_length = only(unique(getindex.(ITensors.params.(smt), :op_length)))
        return !is_id[idx_pos] && op_length > 1
    end

	nt = mapreduce(+, trms_int) do smt
        ops = prime.(map(ITensors.which_op, filter(smt) do op
            site = ITensors.site(op)
            !(site == pos_next)
        end), 1, plev = 1)

		tensor_list = vcat(T, ops)
		opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
		ϕ′ = noprime(contract(tensor_list; sequence = opt_seq))
		return prime(ϕ′, res_inds) * dag(ϕ′)
		#tensor_list = vcat(rho, ops, prime.(dag.(ops)))
        #opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
        #prime(contract(tensor_list; sequence = opt_seq), -2)
    end
	trm_id = filter(envs_trm) do smt
        sites = ITensors.site.(smt)
        is_id = getindex.(ITensors.params.(smt), :is_identity)
        idx_pos = findfirst(isequal(pos_next), sites)
        op_length = only(unique(getindex.(ITensors.params.(smt), :op_length)))
        return !is_id[idx_pos] && op_length == 1
    end
	 # get the number of these operators, rho has to be added weighted by this number
    n_trms_id = length(trm_id)

	rho_lower = prime(Tlower, res_inds) * dag(Tlower)

	rhon = n_trms_id * rho + rho_lower + nt
	return rhon
end




###### VecProj which contains any configuration of ProjTPOs and ProjTTNs ########

struct VecProj{N<:AbstractNetwork, T, P<:Tuple{Vararg{AbstractProjTPO{N, T}}}} <: AbstractProjTPO{N,T}
    net::N
    data::P
    ortho_center::Vector{Int64}
end

function VecProj(all_projs::Tuple)
    ortho_center = all_projs[1].ortho_center
    return VecProj(network(all_projs[1]), all_projs, ortho_center)
end

# maybe updating issue here, we will see
function update_environments!(vecproj::VecProj, isom::ITensor, pos::Tuple{Int,Int}, pos_final::Tuple{Int,Int})
    for (idx,proj) in enumerate(vecproj.data)
        update_environments!(proj, isom, pos, pos_final)
    end
    return vecproj
end

function ∂A(proj_ttn::ProjTTN, pos::Tuple{Int,Int})

    function action(T::ITensor)
        tensor_list = vcat(T, proj_ttn.local_env, dag(prime(proj_ttn.local_env)))
        opt_seq = ITensors.optimal_contraction_sequence(tensor_list)
        return proj_ttn.weight * (noprime(contract(tensor_list; sequence = opt_seq)))
    end

end

function ∂A(proj_operator::VecProj, pos::Tuple{Int,Int})
   
    action_vec = map(ptpo -> ∂A(ptpo, pos), proj_operator.data)

    function action(T::ITensor)
        return mapreduce(+, action_vec) do act
            return act(T)
        end
    end
end
