abstract type AbstractNode{S, I} end

# overload TTNKit functions
TTNKit.spacetype(::Type{<:AbstractNode{S}}) where S = S
TTNKit.spacetype(nd::AbstractNode) = spacetype(typeof(nd))
TensorKit.sectortype(::Type{<:AbstractNode{S,I}}) where {S,I} = I
TensorKit.sectortype(nd::AbstractNode) = sectortype(typeof(nd))

backend(::AbstractNode{Index, I}) where{I} = ITensorsBackend
backend(::AbstractNode{S, I}) where{S,I}   = TensorKitBackend

# getter functions

# defining a hilbertspace by the node space type. may be depricated in future
hilbertspace(nd::AbstractNode, sectors, args...; kwargs...) = spacetype(nd)(sectors, args...; kwargs...)

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
abstract type PhysicalNode{S, I} <: AbstractNode{S,I} end

function Base.show(io::IO, nd::PhysicalNode)
    s = "Node ($(description(nd))), Number: $(position(nd)),"
    s *= " Hilbertspace Type: $(hilbertspace(nd))"
    print(io, s) 
end

# Different from general nodes, physical nodes should implement a valid hilbertspace
# associated to them
hilbertspace(nd::PhysicalNode) = nd.hilbertspace

function _reorder_state(st::Vector, sp::Vector{Pair{Int, Int}})
    perm = sortperm(sp, lt = (x,y) -> isless(x[1],y[1]))

    # check if inside a group of a QN the numbers are increasing
    sp_n = sp[perm]
    idx = 1
    while idx < length(perm)
        q = sp_n[idx][1]
        idx_n = findlast(getindex.(sp_n,1) .== q)
        perm_slice = @view perm[idx:idx_n]
        sort!(perm_slice)
        idx = idx_n+1
    end
    st_n = st[perm]
    return st_n
end

function ITensors.state(nd::PhysicalNode{S,I}, state_str::AbstractString, elT::DataType = ComplexF64) where{S,I}
    st_raw = state(nd, Val(Symbol(state_str)))
    elt_t = promote_type(elT, eltype(st_raw))
    st_raw = elt_t.(st_raw)
    if I == Trivial
        dom = S(1)
    else
        idx_non_zero = findall(!iszero, st_raw)
        sp = space(nd)
        irreps = sp[idx_non_zero]
        if !(all(irreps .== irreps[1]))
            error("Try to set a state with unequal irreps: $(state_str)")
        end
        dom = S(irreps[1][1] => 1)
        st_raw = _reorder_state(st_raw, sp)
    end

    return TensorMap(st_raw, hilbertspace(nd) ← dom)
end


function ITensors.op(nd::PhysicalNode, op_str::AbstractString)
    return op(nd, Val(Symbol(op_str)))
end

#=
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
=#

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

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, node::TTNKit.AbstractNode)
    g = create_group(parent, name)
    g["s"] = node.s
    g["desc"] = node.desc

    if isa(node, TTNKit.PhysicalNode)
        attributes(g)["type"] = "physical"
        write(g, "hilbertspace", node.hilbertspace)
    else
        attributes(g)["type"] = "not-physical"
    end
end

function HDF5.read(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{TTNKit.AbstractNode})
    g = open_group(parent, name)
    s = read(g, "s")
    desc = read(g, "desc")
    read(attributes(g)["type"]) != "physical" && return Node(s, Int64, desc)
    hilberstpace = read(g, "hilbertspace", Index)
    return ITensorNode(s, hilberstpace)
end
