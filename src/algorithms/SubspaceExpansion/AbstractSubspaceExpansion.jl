abstract type AbstractSubspaceExpander end
abstract type NonTrivialExpander <: AbstractSubspaceExpander end

struct NoExpander <: AbstractSubspaceExpander end
expand(T::ITensor, A::ITensor, ::AbstractSubspaceExpander; kwargs...) = T, A



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
    id_pd  = _padding(idfa, id_max, expander.p)
    id_n   = intersect(directsum(idfa, id_pd), id_max; tags = tags(id_sh))


    # correct the direction for the A tensor
    id_dir = dir(inds(A)[end])
    if id_dir != dir(id_n)
        id_n = dag(id_n)
    end
    An = _enlarge_tensor(A, id_au, id_sh, id_n, false)
    # correct the direction for the B tensor
    id_dir = dir(inds(B)[end])
    if id_dir != dir(id_n)
        id_n = dag(id_n)
    end
    Bn = _enlarge_tensor(B, id_bu, id_sh, id_n, true)

    if reorthogonalize
        Bn,R = factorize(Bn, id_bu; tags = tags(id_n))
        An = normalize!(An*R)
    end
    return An, Bn
end