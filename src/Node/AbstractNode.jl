abstract type AbstractNode{S<:IndexSpace, I<:Sector} end

space(::AbstractNode{S}) where S = S
space(::Type{<:AbstractNode{S}}) where S = S
hilbertspace(nd::AbstractNode, sectors) = space(nd)(sectors)
position(nd::AbstractNode) = nd.s
description(nd::AbstractNode) = nd.desc
TensorKit.sectortype(::AbstractNode{S,I}) where{S,I} = I
TensorKit.sectortype(::Type{<:AbstractNode{S,I}}) where {S,I} = I

function hilbertspace(::AbstractNode)
    error("Hilbertspace for non physical nodes only meaningful in combinations with sectors.")
end

function Base.show(io::IO, nd::AbstractNode)
    s = "Node ($(description(nd))), Number: $(position(nd)),"
    s *= " Space Type: $(space(nd))"
    print(io, s) 
end

abstract type PhysicalNode{S<:IndexSpace, I<:Sector} <: AbstractNode{S,I} end

function Base.show(io::IO, nd::PhysicalNode)
    s = "Node ($(description(nd))), Number: $(position(nd)),"
    s *= " Hilbertspace Type: $(hilbertspace(nd))"
    print(io, s) 
end


hilbertspace(nd::PhysicalNode) = nd.hilbertspace


function VirtualNode(nd::AbstractNode, s::Int, desc::AbstractString="")
    return VirtualNode(nd)(s,desc)
end
function VirtualNode(::AbstractNode{S,I}) where{S,I}
    return Node{S,I}
end


function state(nd::PhysicalNode, state_str::AbstractString; elT = ComplexF64)
    hilb = hilbertspace(nd)
    dim_hilb  = dim(hilb)

    sec = _state(typeof(nd), state_str)

    if(sectortype(nd) == Trivial)
        vec = zeros(eltype(sec[2]), dim_hilb)
        vec[sec[1]] = sec[2]
        top_hilb = space(nd)(1)
        return TensorMap(elT.(vec), hilb ← top_hilb)
    end
    top_hilb = space(nd)(sec[1] => 1)
    return sec[2] * TensorMap(ones, elT, hilb ← top_hilb )
end