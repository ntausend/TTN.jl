abstract type AbstractSubspaceExpander end
abstract type NonTrivialExpander <: AbstractSubspaceExpander end

struct NoExpander <: AbstractSubspaceExpander end
expand(T::ITensor, A::ITensor, ::AbstractSubspaceExpander; kwargs...) = T, A
expand(A::ITensor, Chlds::Tuple{ITensor, ITensor}, ::AbstractSubspaceExpander; kwargs...) = A, Chlds...



function truncate_and_move!(ttn::TreeTensorNetwork, A::T, pos::Tuple{Int,Int}, position_next::Union{Tuple{Int,Int}, Nothing},
                              ::AbstractSubspaceExpander; kwargs...) where{T}
    ttn[pos] = A
    !isnothing(position_next) && (ttn = move_ortho!(ttn, position_next))
    return ttn, Spectrum(nothing, 0.0)
end

function truncate_and_move!(ttn::TreeTensorNetwork, A::ITensor, pos::Tuple{Int,Int}, position_next::Union{Tuple{Int,Int}, Nothing},
                              ::NonTrivialExpander; kwargs...)
    if isnothing(position_next)
        ttn[pos] = A
        return ttn, Spectrum(nothing, 0.0)
    end

    net = network(ttn)    
    posnext = connecting_path(net, pos, position_next)[1]

    idx_r = commonind(ttn[pos], ttn[posnext])
    idx_l = uniqueinds(A, idx_r)
    Q,R, spec = factorize(A, idx_l; tags = tags(idx_r), kwargs...)
    #println("Truncated weight: $(1 - sum(eigs(spec)))")

    ttn[pos] = Q
    ttn[posnext] = normalize!(ttn[posnext]*R)

    # orhto direction become undefined here... need to figure out 
    # the index.. however we are not using it anywayrs or?
    #ttn.ortho_direction[pos[1]][pos[2]] 

    ttn.ortho_center .= posnext
    return move_ortho!(ttn, position_next), spec
end

struct DefaultExpander <: NonTrivialExpander
    p::Float64
    min::Int64
    function DefaultExpander(p::Real; min = 1)
        p ≈ 0 && (return NoExpander())
        return new(p, min)
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

    flux_al = flux(Al)
    flux_ar = flux(Ar)
    flux_ac = flux(A)

    # treat the flux as if Ac is contracted to Al
    if !isnothing(flux_al)
        flux_al = flux_ac * dir(id_shl) + flux_al
    end
    
    # again we want the indices to be outgoing
    if hasqns(A)
        id_ual_o = map(i -> sim(i; dir = ITensors.Out), id_ual)
        id_uar_o = map(i -> sim(i; dir = ITensors.Out), id_uar)
        id_shl_o = sim(id_shl; dir = ITensors.Out)
        id_shr_o = sim(id_shr; dir = ITensors.Out)
    else
        id_ual_o = id_ual
        id_uar_o = id_uar
        id_shl_o = id_shl
        id_shr_o = id_shr
    end

    # now tread id_shl as contracted and id_shr for the direction
    #idfal = combinedind(combiner(id_ual; dir = dir(dag(id_shr))))
    #idfar = combinedind(combiner(id_uar; dir = dir(id_shr)))
    idfal = combinedind(combiner(id_ual_o))
    idfar = combinedind(combiner(id_uar_o))

    # now correct the indices to have the correct flux
    #idfal = shift_qn(idfal, flux_al)
    #idfar = shift_qn(idfar, flux_ar)
    if hasqns(idfal)
        idfal = shift_qn(idfal, -flux_al)
        idfar = shift_qn(idfar, -flux_ar)
    end
    # idfal has to be inverted since it build with the inverted direction originally
    #id_maxr = intersect(dag(idfal), idfar)
    id_maxr = intersect(idfal, idfar)
    # the maximal hilbertspace for the left site has to be shifted by the flux of 
    # the central tensor, i.e. redoing the shift of the begining
    # however, this has to be done wiht an arrow inverted according to
    # match our overall flowing scheme
    #id_maxl = combinedind(combiner(id_maxr; dir = dir(dag(id_shr))))
    # direction is inverted, so shift is correct in this way
    #id_maxl = shift_qn(id_maxl, dir(id_shl)*flux_ac)
    id_maxl = shift_qn(id_maxr, flux_ac)



    #id_pdl  = _padding(id_shl, dag(id_maxl), expander.p, expander.min)
    #id_pdr  = _padding(id_shr, id_maxr, expander.p, expander.min)
    id_pdl  = _padding(id_shl_o, id_maxl, expander.p, expander.min)
    id_pdr  = _padding(id_shr_o, id_maxr, expander.p, expander.min)

    #id_nl   = intersect(id_pdl, dag(id_maxl); tags = tags(id_shl))
    id_nl   = intersect(id_pdl, id_maxl; tags = tags(id_shl))
    id_nr   = intersect(id_pdr, id_maxr; tags = tags(id_shr))
    
    if dir(id_nl) != dir(id_shl)
        id_nl = dir(id_nl)
    end
    if dir(id_nr) != dir(id_shr)
        id_nl = dir(id_nr)
    end
    Aln = _enlarge_tensor(Al, id_ual, id_shl, dag(id_nl), false)
    Arn = _enlarge_tensor(Ar, id_uar, id_shr, dag(id_nr), false)
    An = _enlarge_two_leg_tensor(A, (id_nl, id_nr), true)

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
    # get the flux for identifying the different blocks later
    flux_A = flux(_A)
    flux_B = flux(_B)
        
    # permute the tensors to have correct arrangement of indices
    # last index should be the shared index to enlarge
    A = ITensors.permute(_A, id_au..., id_sh)
    B = ITensors.permute(_B, id_bu..., id_sh)

    if hasqns(A)
        # we want all indices to be outgoing for manipulation
        id_au_o = map(i -> sim(i; dir = ITensors.Out), id_au)
        id_bu_o = map(i -> sim(i; dir = ITensors.Out), id_bu)
        id_sh_o = sim(id_sh; dir = ITensors.Out)
    else
        id_au_o = id_au
        id_bu_o = id_bu
        id_sh_o = id_sh
    end
    
    # build the combined indices with the correct direction for hilbertspace arithmetic
    #idfa = combinedind(combiner(id_au; dir = dir(dag(id_sh))))
    #idfb = combinedind(combiner(id_bu; dir = dir(id_sh)))
    idfa = combinedind(combiner(id_au_o))
    idfb = combinedind(combiner(id_bu_o))


    # now correct the indices to have the correct flux
    #idfa = shift_qn(idfa, flux_A)
    #idfb = shift_qn(idfb, flux_B)
    
    if hasqns(idfa)
        idfa = shift_qn(idfa, -flux_A)
        idfb = shift_qn(idfb, -flux_B)
    end
    #@show idfa
    #@show idfb
    #@show id_sh
    # idfa has to be inverted since it build with the inverted direction originally
    id_max = intersect(idfa, idfb)
    #@show id_max

    #id_pd  = _padding(id_sh, id_max, expander.p, expander.min)
    id_pd  = _padding(id_sh_o, id_max, expander.p, expander.min)

    id_n   = intersect(id_pd, id_max; tags = tags(id_sh))

    # id_n should have the correct direction for the A tensor
    if dir(id_n) != dir(id_sh)
        id_n = dag(id_n)
    end
    An = _enlarge_tensor(A, id_au, id_sh, id_n, true)
    # but the B tensor needs to be inverted
    Bn = _enlarge_tensor(B, id_bu, id_sh, dag(id_n), false)

    if reorthogonalize
        Bn,R = factorize(Bn, id_bu; tags = tags(id_n))
        An = normalize!(An*R)
    end
    return An, Bn
end