## Adapt lazy applied operators from ITensors??

abstract type AbstractInteraction{L, N, S<:EuclideanSpace, elT} end

action(int::AbstractInteraction) = int.data
interaction_length(::AbstractInteraction{L}) where L = L
number_of_tensors(::AbstractInteraction{L,N}) where{L,N} = N
space(::AbstractInteraction{L,N,S}) where{L, N,S} = S

interaction_positions(int::AbstractInteraction) = int.positions
coupling(int::AbstractInteraction) = int.coupling
name(int::AbstractInteraction) = int.name

import Base: eltype
eltype(::AbstractInteraction{N,S,elT}) where{N,S,elT} = elT

function Base.show(io::IO, int::AbstractInteraction) 
    println(io, "Interaction $(name(int)) acting on sites: $(interaction_positions(int))")
    println(io, "With coupling: $(coupling(int))")  
    println(io, action(int))
end

struct LocalInteraction{S<:EuclideanSpace, elT} <: AbstractInteraction{0, S, elT}
    data::OnSiteOperator{S}
    coupling::elT
    positions::Int
    name::AbstractString
    function LocalInteraction(int::OnSiteOperator{S}, pos::Int, coupling::elT = 1.0, name::AbstractString = "") where{S, elT}
        elt_n = promote_type(elT, eltype(int))
        return new{S, elt_n}(int, coupling, pos, name)
    end
end


struct TwoSiteInteraction{N, S<:EuclideanSpace, elT} <: AbstractInteraction{N, S, elT}
    data::Tuple{TreeLegTensor{S}, TreeLegTensor{S}}
    coupling::elT
    positions::Tuple{Int, Int}
    name::AbstractString
    function TwoSiteInteraction(ints::Tuple{TreeLegTensor{S}, TreeLegTensor{S}}, pos::Tuple{Int, Int}, 
                                coupling::elT = 1.0, name::AbstractString= "") where{S, elT}
        elt_n = promote_type(elT, eltype.(ints)...)
        return new{abs(pos[2] - pos[1]), S, elt_n}(ints, coupling, pos, name)
    end

end

function TwoSiteInteraction(int1::TreeLegTensor{S}, pos1::Int, int2::TreeLegTensor{S},
                            pos2::Int, coupling::elT=1.0, name::AbstractString = "") where{S,elT}
    return TwoSiteInteraction((int1, int2), (pos1, pos2), coupling, name) 
end
