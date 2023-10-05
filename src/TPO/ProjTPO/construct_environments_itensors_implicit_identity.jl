# calculate the up rg flow for all operator terms
function _up_rg_flow_future(ttn::TreeTensorNetwork, tpo::TPO)
	net = network(ttn)

	# now we want to calculate the upflow up to the ortho position
	oc = ortho_center(ttn)
	# should be orthogonalized
	@assert oc != (-1,-1)
	# forget every extra information stored in the structure of tpo and
	# cast it to a flat array, we need to reconstruct every sum structure
	# later on using the id saved in the tpo operators itself
	trms = convert_cu(vcat(terms.(tpo.data)...), ttn)
    
	# initialize the rg terms similiar to the bottom envs. i.e.
	# for every layer we have a array for every node denoting the upflow of the link
	# operators up to this point
	# structure:
	# first index  -> layer
	# second index -> node
	# third index  -> leg
	# fourth index -> operator flow
	rg_terms    = Vector{Vector{Vector{Vector{Op}}}}(undef, number_of_layers(net))

	# the first layer terms are simply given by the original tpo terms
	rg_terms[1] = map(eachindex(net, 1)) do pp
		chdnds = child_nodes(net, (1, pp))
		map(1:number_of_child_nodes(net, (1, pp))) do nn
			# do the mapping implemented by wladi here
			filter(trm -> isequal(chdnds[nn], site(trm)), trms)
		end
	end

	# now we calculate the upflow of the operator terms, always tracking the original
	# tensor of the sum
	for ll in Iterators.drop(eachlayer(net), 1)
		rg_terms[ll] = Vector{Vector{Vector{Op}}}(undef, number_of_tensors(net, ll))
		for pp in eachindex(net, ll)
			n_chds = number_of_child_nodes(net, (ll, pp))
			rg_terms[ll][pp] = Vector{Vector{Op}}(undef, n_chds)

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
                    # getting the open links which have to be formally multiplied by identities
					open_links::Vector{Index} = map(filter(s -> s ∉ site.(trms), child_nodes(net, chd))) do (ll, pp)
						if ll == 0 # lowest layer, different names
							return only(inds(Tn; tags = "Site,n=$pp"))
						else
							return only(inds(Tn; tags = "Link,nl=$ll,np=$pp"))
						end
					end

                    # contract the non-trivial terms into the rg_operator
                    # prime the indicies where identities appear
                    _rg_op = prime!(reduce(*, _ops, init = Tn), open_links) * dag(prime(Tn))

					# now build the new params list
					prm   = params.(trms)
					# summand index, should be the same for all
					smt   = only(unique(getindex.(prm, :sm)))
					#pdtid = Tuple(vcat(map(p -> vcat(p...), getindex.(prm, :pd))...))
					op_length = only(unique(getindex.(prm, :op_length))) - op_reduction
					# these operators are always non-identity flows
					# i.e. this should evaluate to false
					is_identity = false#all(getindex.(prm, :is_identity))
					return Op(_rg_op, chd; sm = smt, op_length = op_length, is_identity = is_identity)
				end
				# now collapse all pure onsite operators only on this node
				rg_terms[ll][pp][index_of_child(net, chd)] = _collapse_onsite(rg_trms_new)
			end
		end
	end

	return rg_terms
end


function _build_environments_future(ttn::TreeTensorNetwork, rg_flow_trms::Vector{Vector{Vector{Vector{Op}}}})
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
	
	# last environment is simply the product operator of all non-trivial legs
	# legs not appearing have implicit identities acting on them
	environments[end][1] = map(smt -> reduce(*, smt, init = Prod{Op}()), trms)

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
			# index pointing on which we define the new environment
			idx_chld = only(inds(Tn; tags = "Link, nl=$ll, np=$pp"))


			# calculate the rg flow of all terms
			rg_trms = map(trms_filtered) do trm
				op_reduction = length(filter(p -> !p, getindex.(params.(trm), :is_identity)))
				if op_reduction > 0
					# if op_reduction == 0 we only have a flow of identities
					op_reduction -= 1
				end
				_ops = which_op.(trm)
			

				# getting all links which do not appear in the product operator
				# filter out all links not appearing in the operator and not being the current node
				open_links = uniqueinds(Tn, map(_o -> vcat(commoninds(_o, Tn), idx_chld), _ops)...)
			
                _rg_op = prime!(reduce(*, _ops, init = Tn), open_links) * dag(prime(Tn))
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
			
			# fetch all together and collect terms corresponding to the same sumid
			trms = _collect_sum_terms(vcat(trms_below..., residual...))
			env_n = map(trm -> reduce(*, trm; init = Prod{Op}()), trms)
			environments[ll][pp] = _collapse_onsite(env_n)
		end		
	end
	return environments
end
