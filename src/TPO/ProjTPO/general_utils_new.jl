_is_identity(trm::Scaled{ComplexF64, Prod{Op}}) = getindex.(params.(terms(trm)), :is_identity)
_is_identity(trm::Op) = getindex(params(trm), :is_identity)
_op_length(trm::Scaled{ComplexF64, Prod{Op}}) = only(unique(getindex.(params.(terms(trm)), :op_length)))

function _collapse_onsite(tpo::Vector{Scaled{ComplexF64, Prod{Op}}})

	tpo_id = filter(all∘_is_identity, tpo)
	tpo_non_id = filter(!all∘_is_identity, tpo)

	#onsite terms
	otrms    = filter(isone∘_op_length, tpo_non_id)
    residual = vcat(filter(!isone∘_op_length, tpo_non_id)..., tpo_id...)

	 # now splitting the operators along the trivial identity parts and the non-trivial ones
	otrms_split = map(otrms) do trm
		trm_t = terms(trm)
		coef = coefficient(trm)
		coef * only(filter(!_is_identity, trm_t)), filter(_is_identity, trm_t)
	end
	otrms_nonid = getindex.(otrms_split,1)
    otrms_id    = getindex.(otrms_split,2)
	#coefs       = getindex.(otrms_split,3)

	# now getting all unique positions from the onsite terms
    positions_u = unique(site.(argument.(otrms_nonid)))
	# now collapse all operators wich are the same
	#otrms_cllps::Vector{Prod{Op}}
	otrms_cllps = map(positions_u) do pos
        id_collapse = findall(trm -> isequal(pos, site(argument(trm))), otrms_nonid)


        op_non_id = otrms_nonid[id_collapse]

        # collaps the non-trivial operators. Keep the identity of the
        # first only

        op_cllps = mapreduce(+,op_non_id) do trm
			coefficient(trm)*which_op(trm)
		end


		# only using the very first identity, since all others should be
        # using the same
        padding_id = otrms_id[id_collapse[1]]

        # getting the combined sumid
        smid = Tuple(sort(vcat(map(sm -> vcat(sm...), getindex.(params.(op_non_id), :sm))...)))

        # explicitly casting on scaled prod op operator for convinient handling later
        op_n = complex(1.0) * Prod{Op}() * Op(op_cllps, pos; sm = smid, is_identity = false, op_length = 1)

        mapreduce(*, init = op_n, padding_id) do id
            Op(which_op(id), site(id); sm = smid, is_identity = true, op_length = 1)
        end
    end
	return vcat(otrms_cllps..., residual...)
end

# version with unpacked product operator
function _collapse_onsite(tpo::Vector{Vector{Op}}, coef::Vector{ComplexF64})
	# get all terms only composed out of padding identities

	tpo_id = filter(all∘_is_identity, tpo)
	tpo_non_id = filter(!all∘_is_identityl, tpo)
	#onsite terms
	otrms    = filter(isone∘_op_length, tpo_non_id)
	residual = vcat(filter(!isone∘_op_length, tpo_non_id)..., tpo_id...)

	# now splitting the operators along the trivial identity parts and the non-trivial ones
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
	return vcat(otrms_cllps, residual...)
end

# version wich operators only on a vector and collapses all length one operators acting on the same site
# this is unsafe and may cause not correct padding of the identities. Should one search for all identities
# with the same sum id?
function _collapse_onsite(tpo::Vector{Op})
	return vcat(_collapse_onsite(_collect_sum_terms(tpo))...)
end


extract_sm_id(trms::Vector{Op}) = getindex.(params.(trms), :sm)

function _collect_sum_terms(trms::Vector{Op})
	map(unique(extract_sm_id(trms))) do id
		filter(T -> isequal(id, getindex(params(T),:sm)), trms)
	end
end
