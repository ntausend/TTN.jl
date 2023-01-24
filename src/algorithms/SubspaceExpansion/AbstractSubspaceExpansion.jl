abstract type AbstractSubspaceExpander end
abstract type NonTrivialExpander <: AbstractSubspaceExpander end

struct NoExpander <: AbstractSubspaceExpander end
expand(T::ITensor, A::ITensor, ::AbstractSubspaceExpander; kwargs...) = T, A
expand(A::ITensor, Chlds::Tuple{ITensor, ITensor}, ::AbstractSubspaceExpander; kwargs...) = A, Chlds...



function truncate_and_move!(ttn::TreeTensorNetwork, A::T, pos::Tuple{Int,Int}, position_next::Union{Tuple{Int,Int}, Nothing},
                              ::AbstractSubspaceExpander; kwargs...) where{T}
    ttn[pos] = A
    !isnothing(position_next) && (ttn = move_ortho!(ttn, position_next))
    return ttn
end

function truncate_and_move!(ttn::TreeTensorNetwork, A::ITensor, pos::Tuple{Int,Int}, position_next::Union{Tuple{Int,Int}, Nothing},
                              ::NonTrivialExpander; kwargs...)
    if isnothing(position_next)
        ttn[pos] = A
        return ttn
    end

    net = network(ttn)    
    posnext = connecting_path(net, pos, position_next)[1]

    idx_r = commonind(ttn[pos], ttn[posnext])
    idx_l = uniqueinds(A, idx_r)
    Q,R, spec = factorize(A, idx_l; tags = tags(idx_r), kwargs...)
    #println("Truncated weight: $(1 - sum(eigs(spec)))")

    ttn[pos] = Q
    ttn[posnext] = normalize!(ttn[posnext]*R)
    ttn.ortho_center .= posnext
    return move_ortho!(ttn, position_next)
end

struct DefaultExpander{T} <: NonTrivialExpander
    p::T
    function DefaultExpander(p::N) where {N<:Number}
        p ≈ 0 && (return NoExpander())
        return new{N}(p)
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
    idfal = combinedind(combiner(id_ual))
    idfar = combinedind(combiner(id_uar))

    id_max  = intersect(idfal, idfar)
    id_pdl  = _padding(id_shl, id_max, expander.p)
    id_pdr  = _padding(id_shr, id_max, expander.p)
    id_nl   = intersect(id_pdl, id_max; tags = tags(id_shl))
    id_nr   = intersect(id_pdr, id_max; tags = tags(id_shr))
    # now correct directions and enlarge the tensors
    if (id_nl) != dir(inds(Al)[end])
        id_nl = dag(id_nl)
    end
    if (id_nr) != dir(inds(Ar)[end])
        id_nr = dag(id_nr)
    end
    Aln = _enlarge_tensor(Al, id_ual, id_shl, id_nl, false)
    Arn = _enlarge_tensor(Ar, id_uar, id_shr, id_nr, false)
    if (id_nl) != dir(inds(A)[1])
        id_nl = dag(id_nl)
    end
    if (id_nr) != dir(inds(A)[2])
        id_nr = dag(id_nr)
    end
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
        
    # permute the tensors to have correct arrangement of indices
    # last index should be the shared index to enlarge
    A = ITensors.permute(_A, id_au..., id_sh)
    B = ITensors.permute(_B, id_bu..., id_sh)
    
    idfa = combinedind(combiner(id_au))
    idfb = combinedind(combiner(id_bu))

    id_max = intersect(idfa, idfb)
    #id_max = directsum(idfa, idfb)
    id_pd  = _padding(id_sh, id_max, expander.p)

    id_n   = intersect(id_pd, id_max; tags = tags(id_sh))
    #@show id_sh
    #@show id_n


    # correct the direction for the A tensor
    id_dir = dir(inds(A)[end])
    if id_dir != dir(id_n)
        id_n = dag(id_n)
    end
    An = _enlarge_tensor(A, id_au, id_sh, id_n, true)
    # correct the direction for the B tensor
    id_dir = dir(inds(B)[end])
    if id_dir != dir(id_n)
        id_n = dag(id_n)
    end
    Bn = _enlarge_tensor(B, id_bu, id_sh, id_n, false)

    if reorthogonalize
        Bn,R = factorize(Bn, id_bu; tags = tags(id_n))
        An = normalize!(An*R)
    end
    return An, Bn
end