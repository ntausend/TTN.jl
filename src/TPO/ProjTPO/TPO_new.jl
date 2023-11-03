struct TPO{L} <: TTNKit.AbstractTensorProductOperator{L}
    lat::L
	data::Vector{Scaled{ComplexF64,Prod{Op}}}
end

# transforming a formal ampo from ITensors to a vector of product operators
# already performing the scaling and extracting the operators from the hilbertspaces
function TPO(ampo::OpSum, lat::AbstractLattice)
	physidx = siteinds(lat)

	sm_terms = filter(t -> !isapprox(coefficient(t),0), terms(sortmergeterms(ampo)))
    # assert correct qns in the case of qn indices -> todo
    # do algebraic reduction for same tensors etc..

	tpo_sum_type = eltype(sm_terms)
    tpo_sum   = Vector{tpo_sum_type}(undef, length(sm_terms))
	 for (jj, stm) in enumerate(sm_terms)
        # saving coefficient
        #coef_list[jj] = coefficient(stm)
        coef = coefficient(stm)
        # extracting the product part
        #prod_op = map(enumerate(terms(stm))) do (pp, prt)
        prod_op = map(terms(stm)) do prt
            _opstr = which_op(prt)
            # convert the side index to an tuple to be compatible with 1D
            idx = site(prt)
            idx isa Int64 && (idx = Tuple(idx))
            idx_lin = linear_ind(lat, idx)
            _op = op(physidx[idx_lin], _opstr; params(prt)...)
            # do we need the product id here? propl. not
            #return Op(_op, (0, idx_lin); sm = Tuple(jj), pd = Tuple(pp), op_length = length(stm), is_identity = false)
            return Op(_op, (0, idx_lin); sm = Tuple(jj), op_length = length(stm), is_identity = false)
        end

        tpo_sum[jj] =  reduce(*, prod_op, init = coef * Prod{Op}())
    end
	# collapse onsite operators acting on the same link
	return TPO{typeof(lat)}(lat, _collapse_onsite(tpo_sum))
end
