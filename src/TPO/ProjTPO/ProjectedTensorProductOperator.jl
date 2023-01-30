struct ProjTPO{N<:TTNKit.AbstractNetwork, T, B<:AbstractBackend} <: AbstractProjTPO{N,T,B}
    net::N
    tpo::Vector{Prod{Op}}
    ortho_center::Vector{Int64} # tracking the current ortho center

    rg_terms_upwards::Vector{Vector{Vector{Vector{Op}}}}
    #rg_id_upwars::Vector{Vector{Vector{T}}}
    environments::Vector{Vector{Vector{Prod{Op}}}}
end

# general constructor from a formal sum, depending on the case it will 
# generate the itensor or tensormap version of this pTPO
function ProjTPO(ttn::TreeTensorNetwork{N, T}, ampo::Sum{Scaled{C, Prod{Op}}}) where {N,T, C}
    net = network(ttn)
    # first construct the tpo, invovles some conversion and returns
    # a vector (representing the sum) of the product operators.
    tpo = _ampo_to_tpo(ampo, physical_lattice(net))
    # calcualte the uprg flows of the operators and identities
    rg_flw_up, id_up_rg = _up_rg_flow(ttn, tpo)
    # build the environments
    envs = _build_environments(ttn, rg_flw_up, id_up_rg)
    return ProjTPO{N, T, backend(ttn)}(net, tpo, vcat(ortho_center(ttn)...), rg_flw_up, envs)
end

environments(ptpo::ProjTPO, pos::Tuple{Int, Int}) = ptpo.environments[pos[1]][pos[2]]

include("./general_utils.jl")
include("./constructing_projtpo_from_ampo_itensors.jl")


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
    envs = projTPO[pos]
    # extract all components comming from below
    terms_filtered = map(envs) do smt
        filter(T -> ITensors.site(T) != pos_final, ITensors.terms(smt))
    end

    # do the rg flow of the terms w.r.t the isometry
    rg_flw = map(terms_filtered) do smt
        _ops = which_op.(smt)
        tensor_list = [isom, _ops..., dag(prime(isom))]
        opt_seq = optimal_contraction_sequence(tensor_list)
				
		_rg_op = contract(tensor_list; sequence = opt_seq)
        prm   = params.(smt)
		# summand index, should be the same for all
		sid  = only(unique(getindex.(prm, :sm)))
		pdtid = Tuple(vcat(map(p -> vcat(p...), getindex.(prm, :pd))...))
		Op(_rg_op, pos; sm = sid, pd = pdtid)
    end
    # collapse the onsite operators
    # TODO currently disabled... need to restore the information wich operator is Local
    # and which is shared...

    envs_final = projTPO[pos_final]
    # now rebuild the environments for the new node
    projTPO.environments[pos_final[1]][pos_final[2]] = map(rg_flw) do trm
        sid = getindex(params(trm), :sm)
        idxsm = findfirst(envs_final) do smt_fin
            sid == only(unique(extract_sm_id(terms(smt_fin))))
        end

        reduce(*, vcat(filter(T -> site(T) != pos, terms(envs_final[idxsm])), trm), init = Prod{Op}())
    end

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