function _collapse_onsite(tpo::Vector{Prod{Op}})
	# get all terms only composed out of padding identities

	tpo_id = filter(_op -> all(getindex.(params.(_op), :is_identity)),tpo)
	tpo_non_id = filter(_op -> !all(getindex.(params.(_op), :is_identity)),tpo)
	#onsite terms
	op_length(_op::Prod{Op}) = only(unique(getindex.(params.(_op), :op_length)))
	otrms    = filter(isone∘op_length, tpo_non_id)
	residual = vcat(filter(!isone∘op_length, tpo_non_id)..., tpo_id...)

	# now splitting the operators along the trivial identity parts and the non-trivial ones
	is_identity(_op::Op) = getindex(params(_op), :is_identity)
	otrms_split = map(otrms) do trm
		trm_t = terms(trm)
		only(filter(!is_identity, trm_t)), filter(is_identity, trm_t)
	end
	otrms_nonid = map(term -> getindex(term,1), otrms_split)
	otrms_id    = map(term -> getindex(term,2), otrms_split)

	# now getting all unique positions from the onsite terms
	positions_u = unique(site.(otrms_nonid))#unique(map(site_cmps, otrms))
	# now collapse all operators wich are the same
	otrms_cllps::Vector{Prod{Op}} = map(positions_u) do pos
		id_collapse = findall(trm -> isequal(pos, site(trm)), otrms_nonid)
		#trms = filter(trm -> isequal(pos, site_cmps(trm)), otrms_nonid)
		op_non_id = otrms_nonid[id_collapse]

		# collaps the non-trivial operators. Keep the identity of the
		# first only
		op_cllps = mapreduce(which_op, +, op_non_id)
		# only using the very first identity, since all others should be
		# using the same
		padding_id = otrms_id[id_collapse[1]]
		# getting the combined sumid
		smid = Tuple(sort(vcat(map(sm -> vcat(sm...), getindex.(params.(op_non_id), :sm))...)))
		# explicitly casting on product operator for convinient handling later
		op_n = Prod{Op}() * Op(op_cllps, pos; sm = smid, is_identity = false, op_length = 1)
		mapreduce(*, init = op_n, padding_id) do id
			Op(which_op(id), site(id); sm = smid, is_identity = true, op_length = 1)
		end
	end
	return reduce(vcat, [otrms_cllps, residual...])
end

# version with unpacked product operator
function _collapse_onsite(tpo::Vector{Vector{Op}})
	# get all terms only composed out of padding identities

	tpo_id = filter(_op -> all(getindex.(params.(_op), :is_identity)), tpo)
	tpo_non_id = filter(_op -> !all(getindex.(params.(_op), :is_identity)),tpo)
	#onsite terms
	op_length(_op::Vector{Op}) = only(unique(getindex.(params.(_op), :op_length)))
	otrms    = filter(isone∘op_length, tpo_non_id)
	residual = vcat(filter(!isone∘op_length, tpo_non_id)..., tpo_id...)

	# now splitting the operators along the trivial identity parts and the non-trivial ones
	is_identity(_op::Op) = getindex(params(_op), :is_identity)
	otrms_split = map(otrms) do trm
		only(filter(!is_identity, trm)), filter(is_identity, trm)
	end
	otrms_nonid = map(term -> getindex(term,1), otrms_split)
	otrms_id    = map(term -> getindex(term,2), otrms_split)

	positions_u = unique(site.(otrms_nonid))
	# now collapse all operators wich are the same
	otrms_cllps::Vector{Vector{Op}} = map(positions_u) do pos
		id_collapse = findall(trm -> isequal(pos, site(trm)), otrms_nonid)
		#trms = filter(trm -> isequal(pos, site_cmps(trm)), otrms_nonid)
		op_non_id = otrms_nonid[id_collapse]

		# collaps the non-trivial operators. Keep the identity of the
		# first only
		op_cllps = mapreduce(which_op, +, op_non_id)
		# only using the very first identity, since all others should be
		# using the same
		padding_id = otrms_id[id_collapse[1]]
		# getting the combined sumid
		smid = Tuple(sort(vcat(map(sm -> vcat(sm...), getindex.(params.(op_non_id), :sm))...)))
		# explicitly casting on product operator for convinient handling later
		op_n = [Op(op_cllps, pos; sm = smid, is_identity = false, op_length = 1)]
		padding_id = map(padding_id) do id
			Op(which_op(id), site(id); sm = smid, is_identity = true, op_length = 1)
		end
		vcat(op_n, padding_id...)
	end
	return reduce(vcat, [otrms_cllps, residual...])
end

# version wich operators only on a vector and collapses all length one operators acting on the same site
# this is unsafe and may cause not correct padding of the identities. Should one search for all identities
# with the same sum id?
function _collapse_onsite(tpo::Vector{Op})
	return vcat(_collapse_onsite(_collect_sum_terms(tpo))...)
end


extract_sm_id(trms::Vector{Op}) = getindex.(params.(trms), :sm)

function unique_tuples(vec_tuples::Vector{T}) where T<:Tuple
    # Convert each tuple to a vector (enables iterative hashing)
    tuple_to_vec(t::Tuple) = collect(t)
    # Use unique-by-vector to avoid recursive hashing
    unique_tuples = unique(tuple_to_vec, vec_tuples)
    return map(t->(t...,), unique_tuples)
end

function _collect_sum_terms(trms::Vector{Op})
	map(unique_tuples(extract_sm_id(trms))) do id
		return filter(T -> isequal(collect(id), collect(getindex(params(T),:sm))), trms)
	end
end
