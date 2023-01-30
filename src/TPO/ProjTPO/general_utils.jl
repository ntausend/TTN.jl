function _collapse_onsite(tpo_v::Vector{Vector{Op}})
	#onsite terms
	otrms    = only.(filter(isone∘length, tpo_v))
	residual = filter(!isone∘length, tpo_v)
	#otrms_cllps = mapreduce(which_op, +, only.(otrms))
	# now filter the onsite terms according to the site
	positions_u = unique(site.(otrms))
	otrms_cllps = map(positions_u) do pos
		trms = filter(T -> isequal(pos, site(T)), otrms)
		op_cllps = mapreduce(which_op, +, trms)
		smid = Tuple(vcat(map(sm -> [s for s in sm], getindex.(params.(trms), :sm))...) )
		pid = Tuple(vcat(map(sm -> [s for s in sm], getindex.(params.(trms), :pd))...) )
		[Op(op_cllps, pos; sm = smid, pd = pid)]
	end
	return vcat(otrms_cllps, residual)
end

extract_sm_id(trms::Vector{Op}) = getindex.(params.(trms), :sm)

function _collect_sum_terms(trms::Vector{Op})
	map(unique(extract_sm_id(trms))) do id
		filter(T -> isequal(id, getindex(params(T),:sm)), trms)
	end
end