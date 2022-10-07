abstract type AbstractNode{S<:IndexSpace, I<:Sector} end

# overload TTNKit functions
TTNKit.spacetype(::Type{<:AbstractNode{S}}) where S = S
TTNKit.spacetype(nd::AbstractNode) = spacetype(typeof(nd))
TensorKit.sectortype(::Type{<:AbstractNode{S,I}}) where {S,I} = I
TensorKit.sectortype(nd::AbstractNode) = sectortype(typeof(nd))


# getter functions

# defining a hilbertspace by the node space type. may be depricated in future
hilbertspace(nd::AbstractNode, sectors) = spacetype(nd)(sectors)

# linear position of the node
position(nd::AbstractNode) = nd.s
# string description of the node
description(nd::AbstractNode) = nd.desc

# hilbert space of a general abstract node not meaningful since only physical nodes have already
# initialized a hilbertspace.
function hilbertspace(::AbstractNode)
    error("Hilbertspace for non physical nodes only meaningful in combinations with sectors.")
end

function Base.show(io::IO, nd::AbstractNode)
    s = "Node ($(description(nd))), Number: $(position(nd)),"
    s *= " Space Type: $(spacetype(nd))"
    print(io, s) 
end

# abstract physical node assiciated to the physical layer of the tree
abstract type PhysicalNode{S<:IndexSpace, I<:Sector} <: AbstractNode{S,I} end

function Base.show(io::IO, nd::PhysicalNode)
    s = "Node ($(description(nd))), Number: $(position(nd)),"
    s *= " Hilbertspace Type: $(hilbertspace(nd))"
    print(io, s) 
end

# Different from general nodes, physical nodes should implement a valid hilbertspace
# associated to them
hilbertspace(nd::PhysicalNode) = nd.hilbertspace



function state(nd::PhysicalNode, state_str::AbstractString; elT::DataType = ComplexF64)
    hilb = hilbertspace(nd)
    
    st = state_dict(typeof(nd))[state_str]
    elt_t = promote_type(elT, eltype(st))
    st = elt_t.(st)
    if sectortype(nd) == Trivial
        dom = spacetype(nd)(1)
    else
        charge = charge_dict(typeof(nd))[state_str]
        dom = spacetype(nd)(charge => 1)
    end
    return TensorMap(st, hilb ← dom)
end


function ==(nd1::AbstractNode, nd2::AbstractNode)
    spacetype(nd1)  == spacetype(nd2)    || return false
    sectortype(nd1) == sectortype(nd2)   || return false
    position(nd1) == position(nd2)       || return false
    description(nd1) == description(nd2) || return false
    return true
end
function ==(nd1::PhysicalNode, nd2::PhysicalNode)
    spacetype(nd1)  == spacetype(nd2)      || return false
    sectortype(nd1) == sectortype(nd2)     || return false
    position(nd1) == position(nd2)         || return false
    description(nd1) == description(nd2)   || return false
    hilbertspace(nd1) == hilbertspace(nd2) || return false
    return true
end

==(nd1::AbstractNode, nd2::PhysicalNode) = false
==(nd1::PhysicalNode, nd2::AbstractNode) = false