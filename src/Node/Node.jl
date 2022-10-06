struct Node{S<:IndexSpace, I<:Sector} <: AbstractNode{S, I}
    s::Int
    desc::AbstractString
end

# trivial node fast constructor.. may be depricated in the future
function Node(s::Int, desc=""; field::Union{Field, Type{<:EuclideanSpace}} = ComplexSpace)
    hilb = field isa Field ? field^1 : field(1)
    return Node(s, typeof(hilb), desc)
end

Node(ind::Int, S::Type{<:IndexSpace}, desc="") = Node{S, sectortype(S)}(ind, desc)

# returns an (virtual) node copy of the input node with linear position `s`
# and description `desc`, fogets about possible additional structure if `nd`
# is a PhysicalNode
nodetype(nd::AbstractNode{S,I}) where{S,I} = nodetype(typeof(nd))
nodetype(::Type{<:AbstractNode{S,I}}) where{S,I} = Node{S,I}
Node(nd::AbstractNode, s::Int, desc::AbstractString="") = nodetype(nd)(s,desc)

#=
function hilbertspace(nd::Node{S, Trivial}, sectors::Int, maxdim::Int) where{S}
    dm = min(maxdim, sectors)
    return space(nd)(dm)
end
=#

struct TrivialNode{S<:IndexSpace} <: PhysicalNode{S,Trivial}
    s::Int
    hilbertspace::S
    desc::AbstractString
    function TrivialNode(pos::Int, desc::AbstractString="";
                        local_dim::Int = 2,
                        field::Union{Field, Type{<:EuclideanSpace}} = ComplexSpace)
        hilb = field isa Field ? field^local_dim : field(local_dim)
        return new{typeof(hilb)}(pos, hilb, desc)
    end
end