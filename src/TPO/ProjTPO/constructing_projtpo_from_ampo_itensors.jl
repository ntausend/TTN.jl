# transforming a formal ampo from ITensors to a vector of product operators
# already performing the scaling and extracting the operators from the hilbertspaces
function _ampo_to_tpo(ampo::OpSum, lat::AbstractLattice{D, S, I, ITensorsBackend}) where{D,S,I}
	physidx = siteinds(lat)	
    # getting all non-zero terms
    sm_terms = filter(t -> !isapprox(coefficient(t),0), terms(sortmergeterms(ampo)))
    # assert correct qns in the case of qn indices -> todo
	# do algebraic reduction for same tensors etc..
	
    tpo_sum   = Vector{Prod{Op}}(undef, length(sm_terms))

	for (jj, stm) in enumerate(sm_terms)
		# saving coefficient
		#coef_list[jj] = coefficient(stm)
		coef = coefficient(stm)
		# extracting the product part
		prod_op = map(enumerate(terms(stm))) do (pp, prt)
			_opstr = which_op(prt)
			# convert the side index to an tuple to be compatible with 1D
			idx = site(prt)
			idx isa Int64 && (idx = Tuple(idx))
			idx_lin = linear_ind(lat, idx)
			_op = op(physidx[idx_lin], _opstr; params(prt)...)
			return Op(_op, (0, idx_lin); sm = Tuple(jj), pd = Tuple(pp))
		end
		# now rescale the first operator in the list with the coefficient
		prod_op[1] = Op(which_op(prod_op[1])*coef, site(prod_op[1]); params(prod_op[1])...)
		tpo_sum[jj] = reduce(*, prod_op, init = Prod{Op}())
	end
	return tpo_sum
end

# calculate the up rg flow for all operator terms
function _up_rg_flow(ttn::TreeTensorNetwork{N, ITensor}, tpo::Vector{Prod{Op}}) where{N}
	net = network(ttn)

	# now we want to calculate the upflow up to the ortho position
	oc = ortho_center(ttn)
	# should be orthogonalized
	@assert oc != (-1,-1)
	# forget every extra information stored in the structure of tpo and
	# cast it to a flat array, we need to reconstruct every sum structure
	# later on using the id saved in the tpo operators itself
	trms = vcat(terms.(tpo)...)
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
			tmp = filter(trm -> isequal(chdnds[nn], site(trm)), trms)
		end
	end
	id_rg[1] = map(eachindex(net,1)) do pp
		chdnds = child_nodes(net, (1,pp))
		map(1:number_of_child_nodes(net, (1,pp))) do nn
			pos = chdnds[nn][2]
			idx = inds(ttn[(1,pp)], "Site,n=$(pos)")
			delta(dag.(idx), prime.((idx)))
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
					pdtid = Tuple(vcat(map(p -> vcat(p...), getindex.(prm, :pd))...))
					return Op(_rg_op, chd; sm = smt, pd = pdtid)
				end
				# now collapse all pure onsite operators only on this node
				rg_terms[ll][pp][index_of_child(net, chd)] = rg_trms_new

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


function _build_environments(ttn::TreeTensorNetwork{N, ITensor}, 
            rg_flow_trms::Vector{Vector{Vector{Vector{Op}}}}, id_up_rg::Vector{Vector{Vector{ITensor}}}) where{N}
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
	#trms = _collapse_onsite(_collect_sum_terms(vcat(rg_flow_trms[end][1]...)))
	trms = _collect_sum_terms(vcat(rg_flow_trms[end][1]...))
	environments[end][1] = map(trms) do smt
		# get all rg_identites on the missing legs
		open_legs = filter(s -> s ∉ site.(smt), child_nodes(net, (nlayers,1)))
		ids = map(open_legs) do ol
				ii = id_up_rg[nlayers][1][index_of_child(net, ol)]
				Op(ii, ol; sm = getindex(params(smt[1]), :sm), pd = Tuple(-1))
		end
		_ops = vcat(smt, ids)
		reduce(*, _ops, init = Prod{Op}())
	end

    #=
	T = ttn[nlayers,1]
	id_prev_fl = map(1:number_of_child_nodes(net, (nlayers, 1))) do chd
		# extract all rg identities from all other identites
		idid = deleteat!(collect(1:number_of_child_nodes(net, (nlayers,1))), chd)
		tensor_list = vcat(T, id_up_rg[nlayers][1][idid], dag(prime(T)))
		opt_seq = optimal_contraction_sequence(tensor_list)
		idn = contract(tensor_list; sequence = opt_seq)
        return idn
	end
    =#

	# now got backwards through the network and calculate the downflow
	for ll in Iterators.drop(Iterators.reverse(eachlayer(net)), 1)
		environments[ll] = Vector{Vector{Prod{Op}}}(undef, number_of_tensors(net,ll))
		for pp in eachindex(net, ll)
			prnt_nd = parent_node(net, (ll,pp))
			#idx_chd = index_of_child(net, (ll, pp))
			Tn = ttn[prnt_nd]
			# From the enviroments, extract all terms except the
			# current node of interest
			trms = vcat(terms.(environments[prnt_nd[1]][prnt_nd[2]])...)
			
			# collect all terms appearing in one sum
			trms_filtered = _collect_sum_terms(filter(T -> site(T) != (ll,pp), trms))
			
			#==#

			# calculate the rg flow of all terms
			rg_trms = map(trms_filtered) do trm
				
				_ops = which_op.(trm)
				
				tensor_list = [Tn, _ops..., dag(prime(Tn))]
            	opt_seq = optimal_contraction_sequence(tensor_list)
				
				_rg_op = contract(tensor_list; sequence = opt_seq)
				
				prm   = params.(trm)
				# summand index, should be the same for all
				smt   = only(unique(getindex.(prm, :sm)))
				pdtid = Tuple(vcat(map(p -> vcat(p...), getindex.(prm, :pd))...))
				Op(_rg_op, (prnt_nd); sm = smt, pd = pdtid)
			end
			

			# identities comming from above needs to be splitted among
			# their individual sumands. there should only be one
			# identity coming from above
			id_tn = filter(T -> all(getindex(params(T), :pd) .== -1), rg_trms)
			
			# some onsite potential collapsing possible here?
			residual = filter(T -> any(getindex(params(T), :pd) .!= -1), rg_trms)
			id_split = vcat(map(id_tn) do id
						map(vcat(getindex(params(id), :sm)...)) do smid
						pdid = getindex(params(id), :pd)
						Op(which_op(id), site(id); sm = Tuple(smid), pd = (-1,))
					end
			end...)
						
			# now we need to collect all terms from below and eventually
			# some identities on the open legs
			# todo ->  collapse has to be done before padding with identites from
			# above
			trms_bl = _collapse_onsite(rg_flow_trms[ll][pp])
			trms = _collect_sum_terms(vcat(id_split..., residual..., trms_bl...))
			environments[ll][pp] = map(trms) do smt
					# get all rg_identites on the missing legs
					open_legs = filter(s -> s ∉ site.(smt), child_nodes(net, (ll,pp)))
					ids = map(open_legs) do ol
						ii = id_up_rg[ll][pp][index_of_child(net, ol)]
						Op(ii, ol; sm = getindex(params(smt[1]), :sm), pd = Tuple(-1))
					end
					_ops = vcat(smt, ids)
					reduce(*, _ops, init = Prod{Op}())
				end
		end		
	end
	return environments
end