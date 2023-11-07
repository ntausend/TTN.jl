abstract type AbstractSubspaceExpander end
abstract type NonTrivialExpander <: AbstractSubspaceExpander end

struct NoExpander <: AbstractSubspaceExpander end
expand(T::ITensor, A::ITensor, ::AbstractSubspaceExpander; kwargs...) = T, A
expand(A::ITensor, Chlds::Tuple{ITensor, ITensor}, ::AbstractSubspaceExpander; kwargs...) = A, Chlds...

maxiter(expander::NonTrivialExpander) = expander.maxiter
tol(expander::NonTrivialExpander) = expander.tol

replace_nothing(::Nothing, replacement) = replacement
replace_nothing(value, replacement) = value

function update_node_and_move!(ttn::TreeTensorNetwork, A::ITensor, position_next::Union{Tuple{Int,Int}, Nothing}; 
                               normalize = nothing,
                               which_decomp = nothing,
                               mindim = nothing,
                               maxdim = nothing,
                               cutoff = nothing,
                               eigen_perturbation = nothing,
                               svd_alg = nothing)
    

    normalize = replace_nothing(normalize, false)

    @assert is_orthogonalized(ttn)

    pos = ortho_center(ttn)
    # sweep is done, nothing to do
    if isnothing(position_next)
        ttn[pos] = A
        return ttn, Spectrum(nothing, 0.0)
    end

    # otherwise, we need to perform a trunctation/qr decomposition
    net = network(ttn)
    # move towards next node..
    posnext = connecting_path(net, pos, position_next)[1]
    idx_r = commonind(ttn[pos], ttn[posnext])
    idx_l = uniqueinds(A, idx_r)


    Q, R, spec = factorize_own(A, idx_l; tags = tags(idx_r), 
                           mindim,
                           maxdim,
                           cutoff,
                           which_decomp,
                           eigen_perturbation,
                           svd_alg)

    ttn[pos] = Q
    ttn[posnext] = ttn[posnext] * R

    # check if posnext operator only has two links, if yes, we need an additional splitting there
    # somethings not right here.... I don't know but noise and subspace expansion is somehow not working
    # hand in hand, but only for some models...
    #=
    if length(inds(ttn[posnext])) == 2
        T = ttn[pos] * ttn[posnext]
        idx_sh = commonind(ttn[pos], ttn[posnext])
        idx_l = uniqueinds(ttn[pos], idx_sh)
        Q,R = factorize(T, idx_l; tags = tags(idx_sh))
        ttn[pos] = Q
        ttn[posnext] = R
    end
    =#

    normalize && (ttn[posnext] ./= norm(ttn[posnext]))
    ttn.ortho_center .= posnext

    return move_ortho!(ttn, position_next), spec
end

struct DefaultExpander <: NonTrivialExpander
    p::Float64
    min::Int64

    maxiter::Int64
    tol::Number

    function DefaultExpander(p::Real; min = 1, maxiter = 10, tol = 1E-5)
        p ≈ 0 && (return NoExpander())
        return new(p, min, maxiter, tol)
    end
end

# special case of having a two leg tensor. here we need to expand both legs at once to rearange the sectors
function expand(_A::ITensor, _Chlds::Tuple{ITensor, ITensor}, expander::DefaultExpander; reorthogonalize = true)
    @assert hasqns(_A) == all(hasqns.(_Chlds))
    @assert length(inds(_A)) == 2

    Al,Ar = _Chlds
    id_shl = commonind(_A, Al)
    id_shr = commonind(_A, Ar)

    id_ual = uniqueinds(Al, id_shl)
    id_uar = uniqueinds(Ar, id_shr)

    A  = ITensors.permute(_A, id_shl, id_shr)
    Al = ITensors.permute(Al, id_ual..., id_shl)
    Ar = ITensors.permute(Ar, id_uar..., id_shr)

    # treat the flux as if Ac is contracted to Al
    if hasqns(A)
        qn_link_ac = Index([flux(A) => 1], "QNAC"; dir = ITensors.In)
        qn_link_al = Index([flux(Al) => 1], "QNAL"; dir = ITensors.In)
        qn_link_ar = Index([flux(Ar) => 1], "QNAR"; dir = ITensors.In)
        #flux_al = flux_ac * dir(id_shl) + flux_al
    else
        qn_link_ac = Index(1, "QNAC")
        qn_link_al = Index(1, "QNAL")
        qn_link_ar = Index(1, "QNAR")
    end

    # build the combined index, put the flux of the center node on the
    # left index
    idfal  = combinedind(combiner(id_ual..., qn_link_al, qn_link_ac; dir = dir(dag(id_shl))))
    idfar  = combinedind(combiner(id_uar..., qn_link_ar; dir = dir(id_shr)))
    # this is directly the correct right right link,
    id_maxr = intersect(dag(idfal), idfar)
    # left link has to be corrected again, by fusing with the inverse link
    id_maxl = dag(combinedind(combiner(id_maxr, qn_link_ac; dir = dir(dag(id_shl)))))

    id_pdl  = _padding(id_shl, id_maxl, expander.p, expander.min)
    id_pdr  = _padding(id_shr, id_maxr, expander.p, expander.min)
    

    id_nl   = intersect(id_pdl, id_maxl; tags = tags(id_shl))
    id_nr   = intersect(id_pdr, id_maxr; tags = tags(id_shr))
    Aln = _enlarge_tensor(Al, id_ual, id_shl, dag(id_nl), true)
    Arn = _enlarge_tensor(Ar, id_uar, id_shr, dag(id_nr), false)
    An = _enlarge_two_leg_tensor(A, (id_nl, id_nr), false)
    if reorthogonalize
        Aln,Rl = factorize(Aln, id_ual; tags = tags(id_nl))
        Arn,Rr = factorize(Arn, id_uar; tags = tags(id_nr))
        An = normalize!((An*Rl)*Rr)
    end
    return An, Aln, Arn
end

function expand(_A::ITensor, _B::ITensor, expander::DefaultExpander; reorthogonalize = true)
    @assert hasqns(_A) == hasqns(_B)

    id_sh = commonind(_A,_B)
    isnothing(id_sh) && error("A and B tensors do not share a link to expand.")
        
    id_au = uniqueinds(_A, id_sh)
    id_bu = uniqueinds(_B, id_sh)
        
    # permute the tensors to have correct arrangement of indices
    # last index should be the shared index to enlarge
    A = ITensors.permute(_A, id_au..., id_sh)
    B = ITensors.permute(_B, id_bu..., id_sh)

    # build qn link for haveing the qns correctly handled
    if hasqns(A)
        qn_link_a = Index([flux(A) => 1], "QNA"; dir = ITensors.In)
        qn_link_b = Index([flux(B) => 1], "QNB"; dir = ITensors.In)
    else
        qn_link_a = Index(1, "QNA")
        qn_link_b = Index(1, "QNB")
    end
    

    # again build the combined index along the direction of the shared index
    idfa = combinedind(combiner(id_au..., qn_link_a; dir = dir(dag(id_sh))))
    idfb = combinedind(combiner(id_bu..., qn_link_b; dir = dir(id_sh)))
    id_max = intersect(dag(idfa), idfb)
    

    id_pd  = _padding(id_sh, id_max, expander.p, expander.min)

    id_n   = intersect(id_pd, id_max; tags = tags(id_sh))
    
    # id_n should have the correct direction for the A tensor
    An = _enlarge_tensor(A, id_au, id_sh, id_n, false)
    # but the B tensor needs to be inverted
    Bn = _enlarge_tensor(B, id_bu, id_sh, dag(id_n), true)

    if reorthogonalize
        Bn,R = factorize(Bn, id_bu; tags = tags(id_n))
        An = normalize!(An*R)
    end
    return An, Bn
end
