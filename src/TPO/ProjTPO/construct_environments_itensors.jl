# calculate the up rg flow for all operator terms
function _up_rg_flow(ttn::TreeTensorNetwork, tpo::TPO)
	net = network(ttn)

	# now we want to calculate the upflow up to the ortho position
	oc = ortho_center(ttn)
	# should be orthogonalized
	@assert oc != (-1,-1)
	# forget every extra information stored in the structure of tpo and
	# cast it to a flat array, we need to reconstruct every sum structure
	# later on using the id saved in the tpo operators itself
	trms = vcat(terms.(tpo.data)...)
	trms = convert_cu(trms,ttn)
	# initialize the rg terms similiar to the bottom envs. i.e.
	# for every layer we have a array for every node denoting the upflow of the link
	# operators up to this point
	# structure:
	# first index  -> layer
	# second index -> node
	# third index  -> leg
	# fourth index -> operator flow
	rg_terms    = Vector{Vector{Vector{Vector{Op}}}}(undef, number_of_layers(net))
	# flow of the identity operator
	id_rg 		= Vector{Vector{Vector{ITensor}}}(undef, number_of_layers(net))

	
	# the first layer terms are simply given by the original tpo terms
	rg_terms[1] = map(eachindex(net, 1)) do pp
		chdnds = child_nodes(net, (1, pp))
		map(1:number_of_child_nodes(net, (1, pp))) do nn
			# do the mapping implemented by wladi here
			filter(trm -> isequal(chdnds[nn], site(trm)), trms)
		end
	end
	id_rg[1] = map(eachindex(net,1)) do pp
		chdnds = child_nodes(net, (1,pp))
		map(1:number_of_child_nodes(net, (1,pp))) do nn
			pos = chdnds[nn][2]# full overlap where ttn2 replaces the internal tensor with the given tensor T at position pos
			function _inner(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork, T::ITensor, update_site::Tuple{Int,Int})
			
				net = network(ttn1)
			
				#top_pos = (TTNKit.number_of_layers(net),1)
				#move_ortho!(ttn1, top_pos)
				#move_ortho!(ttn2, top_pos)
			
				# contruct the network starting from the first layer upwards
				#ns = number_of_sites(net)
				
				phys_lat = physical_lattice(net)
				if T == TensorMap
					res = map(phys_lat) do (nd)
						isomorphism(hilbertspace(nd), hilbertspace(nd))
					end
				else
					res = map(phys_lat) do nd
						delta(dag.(hilbertspace(nd)), prime(hilbertspace(nd)))
					end
				end
			
			
				for ll in eachlayer(net)
					nt = number_of_tensors(net,ll)
					res_new = Vector{ITensor}(undef, nt)
					for pp in eachindex(net, ll)
						childs_idx = getindex.(child_nodes(net, (ll,pp)),2)
						tn1 = ttn1[ll,pp]
						tn2 = (ll,pp) == update_site ? T : ttn2[ll,pp]
						rpre1 = res[childs_idx[1]]
						rpre2 = res[childs_idx[2]]
						res_new[pp] = _dot_inner(tn1, tn2, rpre1, rpre2)
						#println("At layer $ll and position $pp with result ", res_new[pp])
					end
					res = res_new
				end
				# better exception
				length(res) == 1 || error("Tree Tensor Contraction don't leed to a single resulting tensor.")
				res = res[1]
				sres = ITensors.scalar(res)
			
				return sres
			end
			idx = inds(ttn[(1,pp)], "Site,n=$(pos)")
			convert_cu(dense(delta(dag.(idx), prime.((idx)))), ttn)
			# adapt(CuArray, delta(dag.(idx), prime.((idx))))
			# delta(dag.(idx), prime.((idx)))
		end
	end
	

	# now we calculate the upflow of the operator terms, always tracking the original
	# tensor of the sum
	for ll in Iterators.drop(eachlayer(net), 1)
		rg_terms[ll] = Vector{Vector{Vector{Op}}}(undef, number_of_tensors(net, ll))
		id_rg[ll]    = Vector{Vector{ITensor}}(undef, number_of_tensors(net, ll))
		for pp in eachindex(net, ll)
			n_chds = number_of_child_nodes(net, (ll, pp))
			rg_terms[ll][pp] = Vector{Vector{Op}}(undef, n_chds)
			id_rg[ll][pp]    = Vector{ITensor}(undef, n_chds)

			for chd in child_nodes(net, (ll, pp))
				Tn = ttn[chd]

				rg_trms_chld = vcat(map(child_nodes(net, chd)) do cc
					rg_terms[chd[1]][chd[2]][index_of_child(net,cc)]
				end...)
				# now group together all interaction terms corresponding
				# to the same sumand in the original hamiltonian
				rg_trms_chld = _collect_sum_terms(rg_trms_chld)
				# now we need to go through all terms calculating the upflow
				# of these terms
				rg_trms_new = map(rg_trms_chld) do trms
					op_reduction = length(filter(p -> !p, getindex.(params.(trms), :is_identity)))
					if op_reduction > 0
						# if op_reduction == 0 we only have a flow of identities
						op_reduction -= 1
					end
					# getting the interaction terms
					_ops = which_op.(trms)
					open_links = filter(s -> s ∉ site.(trms), child_nodes(net, chd))
					# get the ids_rg of the open links
					ids = map(open_links) do ol
						id_rg[chd[1]][chd[2]][index_of_child(net, ol)]
					end
					_ops = vcat(_ops, ids)
					
					# getting the up pointing index -> remove this by
					# including the id rg flow?
					tensor_list = [Tn, _ops..., dag(prime(Tn))]
          opt_seq = optimal_contraction_sequence(tensor_list)
          _rg_op = contract(tensor_list; sequence = opt_seq)
					# now build the new params list
					prm   = params.(trms)
					# summand index, should be the same for all
					smt   = only(unique(getindex.(prm, :sm)))
					#pdtid = Tuple(vcat(map(p -> vcat(p...), getindex.(prm, :pd))...))
					op_length = only(unique(getindex.(prm, :op_length))) - op_reduction
					# these operators are always non-identity flows
					# i.e. this should evaluate to false
					is_identity = all(getindex.(prm, :is_identity))
					return Op(_rg_op, chd; sm = smt, op_length = op_length, is_identity = is_identity)
				end
				# now collapse all pure onsite operators only on this node
				rg_terms[ll][pp][index_of_child(net, chd)] = _collapse_onsite(rg_trms_new)

				# now we need to calculate the new rg_ids
				#idx_up = commonind(ttn[(ll,pp)],Tn)
				idchlds = id_rg[chd[1]][chd[2]]
				tensor_list = vcat(Tn, idchlds..., dag(prime(Tn)))
				opt_seq = optimal_contraction_sequence(tensor_list)
				idn = contract(tensor_list; sequence = opt_seq)
				id_rg[ll][pp][index_of_child(net, chd)] = idn
			end
		end
	end

	return rg_terms, id_rg
end


function _build_environments(ttn::TreeTensorNetwork, rg_flow_trms::Vector{Vector{Vector{Vector{Op}}}}, id_up_rg::Vector{Vector{Vector{ITensor}}})
	net = network(ttn)
	nlayers = number_of_layers(net)

	# these are the environments for each node of the lattice
	# as such the indices are ordered as follows:
	# first index  -> layer
	# second index -> node
	# third index  -> summand in the tpo opertor
	# the elements are stored as Prod{Op} to have informations as compact as
	# possible.
	environments = Vector{Vector{Vector{Prod{Op}}}}(undef, nlayers)
	# rg flow of the identities down through the top node -> use this more
	# efficiently?
	#id_rg_dn     = Vector{Vector{ITensor}}(undef, nlayers)
	
	# last layer is simple since it only has the components comming from below
	environments[end] = Vector{Vector{Prod{Op}}}(undef, 1)
	trms = _collect_sum_terms(vcat(rg_flow_trms[end][1]...))
	environments[end][1] = map(trms) do smt
		# get all rg_identites on the missing legs
		open_legs = filter(s -> s ∉ site.(smt), child_nodes(net, (nlayers,1)))
		ids = map(open_legs) do ol
				ii = id_up_rg[nlayers][1][index_of_child(net, ol)]
				smid = only(unique(getindex.(params.(smt), :sm)))
				op_length = only(unique(getindex.(params.(smt), :op_length)))
				Op(ii, ol; sm = smid, is_identity = true, op_length = op_length)
		end
		_ops = vcat(smt, ids)
		reduce(*, _ops, init = Prod{Op}())
	end

	# now got backwards through the network and calculate the downflow
	for ll in Iterators.drop(Iterators.reverse(eachlayer(net)), 1)
		environments[ll] = Vector{Vector{Prod{Op}}}(undef, number_of_tensors(net,ll))
		for pp in eachindex(net, ll)
			prnt_nd = parent_node(net, (ll,pp))
			Tn = ttn[prnt_nd]
			# From the enviroments, extract all terms except the
			# current node of interest
			trms = vcat(terms.(environments[prnt_nd[1]][prnt_nd[2]])...)
			
			# collect all terms appearing in one sum
			trms_filtered = _collect_sum_terms(filter(T -> site(T) != (ll,pp), trms))

			# calculate the rg flow of all terms
			rg_trms = map(trms_filtered) do trm
				op_reduction = length(filter(p -> !p, getindex.(params.(trm), :is_identity)))
				if op_reduction > 0
					# if op_reduction == 0 we only have a flow of identities
					op_reduction -= 1
				end
				_ops = which_op.(trm)
				
				tensor_list = [Tn, _ops..., dag(prime(Tn))]
            	opt_seq = optimal_contraction_sequence(tensor_list)
				_rg_op = contract(tensor_list; sequence = opt_seq)
				
				prm   = params.(trm)
				# summand index, should be the same for all
				smt   = only(unique(getindex.(prm, :sm)))
				op_length = only(unique(getindex.(prm, :op_length))) - op_reduction
				is_identity = all(getindex.(prm, :is_identity))
				#pdtid = Tuple(vcat(map(p -> vcat(p...), getindex.(prm, :pd))...))
				Op(_rg_op, (prnt_nd); sm = smt, is_identity = is_identity, op_length = op_length)
			end
			
			# some onsite potential collapsing possible here?
			residual = filter(T -> !getindex(params(T), :is_identity), rg_trms)
			trms_below = rg_flow_trms[ll][pp]
			# identities comming from above needs to be splitted among
			# their individual sumands as they appear from below. 
			# there should only be one
			# identity coming from above
			id_tn = rg_trms[findfirst(T -> getindex(params(T), :is_identity), rg_trms)]
			
			
			# fetch all together and collect terms corresponding to the same sumid
			trms = _collect_sum_terms(vcat(trms_below..., residual...))
			env_n = map(trms) do trm
				# find all ids from above for each terms
				smidtrm = only(unique(getindex.(params.(trm), :sm)))

				op_length = maximum(unique(getindex.(params.(trm), :op_length)))
				# correct all operator lengths of the terms, since operators flowing up
				# loses one length
				trm_cr = map(trm) do T
					Op(which_op(T), site(T); sm = smidtrm, op_length = op_length, is_identity = getindex(params(T),:is_identity))
				end

				# now get all identities from below for the open legs
				open_legs = filter(s -> s ∉ site.(trm_cr), child_nodes(net, (ll,pp)))
				id_bl = map(open_legs) do ol
					ii = id_up_rg[ll][pp][index_of_child(net, ol)]
					Op(ii, ol; sm = smidtrm, is_identity = true, op_length = op_length)
				end
				# also pad the identity from above it is missing
				if parent_node(net, (ll,pp)) ∉ site.(trm_cr)
					id_up = Op(which_op(id_tn), site(id_tn); sm = smidtrm, is_identity = true, op_length = op_length)
					_ops  = vcat(trm_cr, id_bl..., id_up)
				else
					_ops  = vcat(trm_cr, id_bl...)
				end

				#_ops = vcat(trm, id_up..., id_bl...)
				
				reduce(*, _ops, init = Prod{Op}())
			end
			environments[ll][pp] = _collapse_onsite(env_n)
		end		
	end
	return environments
end













####### ProjTTN ########

function bottom_overlap_environments(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork)
	net = network(ttn1)

	# now we want to calculate the upflow up to the ortho position
	oc = ortho_center(ttn1)
	# should be orthogonalized
	@assert oc != (-1,-1)
	@assert oc == ortho_center(ttn2)

	# initialize the rg terms similiar to the bottom envs. i.e.
	# for every layer we have a array for every node denoting the upflow of the link
	# operators up to this point
	# structure:
	# first index  -> layer
	# second index -> node
	# third index  -> leg
	# fourth index -> ovelap flow

	# flow of the identity operator
	id_rg = Vector{Vector{Vector{ITensor}}}(undef, number_of_layers(net))
	
	# the first layer terms, ttn2 is the daggered state
	id_rg[1] = map(eachindex(net,1)) do pp
		chdnds = child_nodes(net, (1,pp))
		Tn2 = ttn2[(1,pp)]
		map(1:number_of_child_nodes(net, (1,pp))) do nn
			pos = chdnds[nn][2]
			idx1 = inds(ttn1[(1,pp)], "Site,n=$(pos)")
			idx2 = inds(ttn2[(1,pp)], "Site,n=$(pos)")
			delta(dag.(idx1), prime.(idx2))
		end
	end

	# now we calculate the upflow of the overlaps
	for ll in Iterators.drop(eachlayer(net), 1)
		id_rg[ll]    = Vector{Vector{ITensor}}(undef, number_of_tensors(net, ll))
		for pp in eachindex(net, ll)
			n_chds = number_of_child_nodes(net, (ll, pp))
			id_rg[ll][pp]    = Vector{ITensor}(undef, n_chds)

			for chd in child_nodes(net, (ll, pp))
				Tn1 = ttn1[chd]
				Tn2 = ttn2[chd]

				idchlds = id_rg[chd[1]][chd[2]]
				tensor_list = vcat(Tn1,idchlds..., prime(dag(Tn2)))
				opt_seq = optimal_contraction_sequence(tensor_list)
				id_rg[ll][pp][index_of_child(net, chd)] = contract(tensor_list; sequence = opt_seq)
			end
		end
	end

	return id_rg
end

function top_overlap_environments(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork, id_up_overlap::Vector{Vector{Vector{ITensor}}})
	net = network(ttn1)
	nlayers = number_of_layers(net)

	# these are the environments for each node of the lattice
	# as such the indices are ordered as follows:
	# first index  -> layer
	# second index -> node
	# third index  -> overlap flow downwards
	# top layer has no top environment and next-to-top layer top environment has no previous top environment
	top_environments = Vector{Vector{ITensor}}(undef, nlayers-1)
	
	# not sure how to do this for arbitrary number of legs, assume for arb legs only single bottom env is needed
	top_environments[end] = Vector{ITensor}(undef, number_of_tensors(net,nlayers-1))
	for pp in eachindex(net, nlayers-1)
		Tn1 = ttn1[(nlayers,1)]
		Tn2 = ttn2[(nlayers,1)]

		bottom_env_element = id_up_overlap[nlayers][1][pp]
		tensor_list = vcat(Tn1, bottom_env_element, prime(dag(Tn2)))
		#opt_seq = optimal_contraction_sequence(tensor_list)
		top_environments[end][isodd(pp)+1] = contract(tensor_list)#contract(tensor_list; sequence = opt_seq)
	end
	
	# now go backwards through the network and calculate the downflow
	for ll in Iterators.drop(Iterators.reverse(collect(Iterators.drop(eachlayer(net),1))), 1)
		top_environments[ll-1] = Vector{ITensor}(undef, number_of_tensors(net,ll-1))
		for pp in eachindex(net, ll)
			
			prnt_node = (ll,pp)
			Tn1 = ttn1[prnt_node]
			Tn2 = ttn2[prnt_node]
			previous_top_env = top_environments[ll][pp]
			# maybe speed-up by contracting these first
			
			for (idx,chd) in enumerate(child_nodes(net, prnt_node))
				# boolean here is slow, find some faster implementation, not general for arbitrary number of legs
				bottom_env_element = id_up_overlap[ll][pp][isodd(idx)+1]

				# order of tensor list is previous_top_env, ket tensor, bottom_env_element, bra tensor
				tensor_list = vcat(previous_top_env, Tn1, bottom_env_element, prime(dag(Tn2)))
				opt_seq = optimal_contraction_sequence(tensor_list)
				top_environments[ll-1][chd[2]] = contract(tensor_list; sequence = opt_seq)
			end
		end		
	end
	return top_environments
end

function top_overlap_environments(ttn1::TreeTensorNetwork,ttn2::TreeTensorNetwork)
	bottom_envs = bottom_overlap_environments(ttn1, ttn2)
	return top_overlap_environments(ttn1, ttn2, bottom_envs)
end

function build_single_overlap_environment(top_envs::Vector,bottom_envs::Vector,ttn2::TreeTensorNetwork,net::AbstractNetwork,which_site::Tuple{Int,Int})
	
	# different for top layer
	nlayers = number_of_layers(net)
	if which_site[1] == nlayers
		return contract(bottom_envs[nlayers][1]..., dag(prime(ttn2[which_site])))
	end

	# likely more efficient way to do this
	local_envs = Vector{ITensor}(undef, number_of_child_nodes(net,which_site)+1)
	local_envs[1] = top_envs[which_site[1]][which_site[2]] * dag(prime(ttn2[which_site]))
	for chd in child_nodes(net,which_site)
		local_envs[index_of_child(net,chd)+1] = bottom_envs[which_site[1]][which_site[2]][index_of_child(net,chd)]
	end
	return contract(local_envs...)
end

build_single_overlap_environment(top_envs::Vector,bottom_envs::Vector,ttn2::TreeTensorNetwork,which_site::Tuple{Int,Int}) = build_single_overlap_environment(top_envs,bottom_envs,ttn2,network(ttn2),which_site)
build_single_overlap_environment(top_envs::Vector,bottom_envs::Vector,ttn2::TreeTensorNetwork,which_site::Vector{Int}) = build_single_overlap_environment(top_envs,bottom_envs,ttn2,network(ttn2),(which_site[1],which_site[2]))

function build_overlap_environments(top_envs::Vector,bottom_envs::Vector,ttn2::TreeTensorNetwork)
	
	net = network(ttn2)
	nlayers = length(bottom_envs)
	# first index  -> layer
	# second index -> node
	# value  -> product of top_env, bottom_envs left to right
	envs = Vector{Vector{ITensor}}(undef, nlayers)

	# top layer has no top environment, only bottom top_environments
	envs[end] = Vector{Vector{ITensor}}(undef, 1)
	envs[end][1] = contract(bottom_envs[nlayers][1]...)

	# child nodes at layer 2 are weird
	for ll in Iterators.drop(Iterators.reverse(eachlayer(net)),1)
		envs[ll] = Vector{ITensor}(undef, number_of_tensors(net,ll))
		for pp in eachindex(net,ll)
			envs[ll][pp] = build_single_overlap_environment(top_envs,bottom_envs,ttn2,(ll,pp))
		end
	end

	return envs
end

function build_overlap_environments(ttn1::TreeTensorNetwork, ttn2::TreeTensorNetwork)
	bottom_envs = bottom_overlap_environments(ttn1, ttn2)
	top_envs = top_overlap_environments(ttn1, ttn2, bottom_envs)

	return build_overlap_environments(top_envs, bottom_envs,ttn2)
end

# need a function to efficiently recalculate environments for new ortho_center

mutable struct ProjTTN{N<:TTNKit.AbstractNetwork, T} <: AbstractProjTPO{N,T}
	ortho_center::Vector{Int}
	weight::Float64

	psi::TreeTensorNetwork
	psi_overlap::TreeTensorNetwork

	bottom_envs::Vector{Vector{Vector{ITensor}}}
	top_envs::Vector{Vector{ITensor}}

	local_env::ITensor
end

network(projttn::ProjTTN) = network(projttn.psi)

move_ortho!(psi::TreeTensorNetwork,oc::Vector{Int}) = move_ortho!(psi,(oc[1],oc[2]))

function precheck_projttn(psi::TreeTensorNetwork,ttn_orthogonal::TreeTensorNetwork)
	@assert psi.ortho_center == ttn_orthogonal.ortho_center
end

# calculates all top and bottom overlap environments from scratch for a given TTN and the state it is being overlapped with
# the default weight for this overlap is 100.0
function initialize_projttn(psi::TreeTensorNetwork{N,T},ttn_orthogonal::TreeTensorNetwork,weight::Float64=100.0) where{N,T}
	
	# need to have some checks here at the beginning
	precheck_projttn(psi,ttn_orthogonal)
	#println("Prechecks completed")

	oc = ttn_orthogonal.ortho_center
	bottom_envs = bottom_overlap_environments(psi, ttn_orthogonal)
	#println("Bottom environments calculated")
	top_envs = top_overlap_environments(psi, ttn_orthogonal, bottom_envs)
	#println("Top environments calculated")
	local_env = build_single_overlap_environment(top_envs, bottom_envs, ttn_orthogonal, oc)
	#println("Local environment calculated")
	return ProjTTN{N,T}(oc, weight, psi, ttn_orthogonal, bottom_envs, top_envs, local_env)
end

# iteratively generates overlap environments for all states in ortho_states from scratch
function initialize_projttn(psi::TreeTensorNetwork,ortho_states::Vector,weights::Vector{Float64}=fill(100.0,length(ortho_states)))
	all_projttns = Vector{ProjTTN}(undef, length(ortho_states))
	for (idx,ttn2) in enumerate(ortho_states)
		all_projttns[idx] = initialize_projttn(psi,ttn2,weights[idx])
	end
	return all_projttns
end

ProjTTN(psi::TreeTensorNetwork,ortho_states::Vector,weights::Vector{Float64}) = initialize_projttn(psi,ortho_states,weights)
ProjTTN(psi::TreeTensorNetwork,ttn_orthogonal::TreeTensorNetwork,weight::Float64) = initialize_projttn(psi,ttn_orthogonal,weight)
ProjTTN(psi::TreeTensorNetwork,ortho_states::Vector) = initialize_projttn(psi,ortho_states)
ProjTTN(psi::TreeTensorNetwork,ttn_orthogonal::TreeTensorNetwork) = initialize_projttn(psi,ttn_orthogonal)


function build_single_overlap_environment(projttn::ProjTTN,which_site::Tuple{Int,Int})
	return build_single_overlap_environment(projttn.top_envs,projttn.bottom_envs,projttn.psi_overlap,network(projttn),which_site)
end

function update_environments_down!(projttn::ProjTTN, isom::ITensor, pos::Tuple{Int, Int}, pos_final::Tuple{Int, Int})

	#println("Moving from ",pos," to ",pos_final)

	nlayers = number_of_layers(network(projttn))

	tensor_list = Vector{ITensor}(undef, 3 + (pos[1] != nlayers))
	pos[1] != nlayers ? tensor_list[end] = projttn.top_envs[pos[1]][pos[2]] : nothing

	chlds = child_nodes(network(projttn), pos)
	bottom_env_initial = projttn.bottom_envs[pos[1]][pos[2]][findfirst(x -> x != pos_final, chlds)]
	tensor_list[2] = bottom_env_initial

	tensor_list[3] = dag(prime(projttn.psi_overlap[pos]))

	tensor_list[1] = isom

	opt_seq = optimal_contraction_sequence(tensor_list)

	top_env_final = contract(tensor_list; sequence = opt_seq)

	projttn.top_envs[pos_final[1]][pos_final[2]] = top_env_final
	projttn.local_env = build_single_overlap_environment(projttn.top_envs,projttn.bottom_envs,projttn.psi_overlap,pos_final)

	return projttn
end

function update_environments_up!(projttn::ProjTTN, isom::ITensor, pos::Tuple{Int, Int}, pos_final::Tuple{Int, Int})

	#println("Moving from ",pos," to ",pos_final)

	nlayers = number_of_layers(network(projttn))

	tensor_list = Vector{ITensor}(undef, 4)

	bottom_envs_initial = projttn.bottom_envs[pos[1]][pos[2]]
	tensor_list[2] = bottom_envs_initial[1]
	tensor_list[3] = bottom_envs_initial[2]

	tensor_list[4] = dag(prime(projttn.psi_overlap[pos]))
	tensor_list[1] = isom

	opt_seq = optimal_contraction_sequence(tensor_list)
	bottom_env_final = contract(tensor_list; sequence = opt_seq)

	chlds = child_nodes(network(projttn), pos_final)
	projttn.bottom_envs[pos_final[1]][pos_final[2]][findfirst(x -> x == pos, chlds)] = bottom_env_final

	projttn.local_env = build_single_overlap_environment(projttn.top_envs,projttn.bottom_envs,projttn.psi_overlap,pos_final)

	return projttn

end

function update_environments!(projttn::ProjTTN, isom::ITensor, pos::Tuple{Int, Int}, pos_final::Tuple{Int, Int})
	
	# pos_final has to be either a child node or the parent node of pos
    @assert pos_final ∈ vcat(child_nodes(network(projttn), pos), parent_node(network(projttn), pos))

	# move the ortho_center of overlap state to match psi
	move_ortho!(projttn.psi_overlap, (projttn.psi.ortho_center[1],projttn.psi.ortho_center[2]))

	if pos_final in child_nodes(network(projttn), pos)
		# in this case we are moving down the tree
		update_environments_down!(projttn, isom, pos, pos_final)
	elseif pos_final == parent_node(network(projttn), pos)
		# in this case we are moving up the tree
		update_environments_up!(projttn, isom, pos, pos_final)
	else
		error("Invalid final position")
	end

end
