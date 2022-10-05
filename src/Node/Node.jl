struct Node{S<:IndexSpace, I<:Sector} <: AbstractNode{S, I}
    s::Int
    desc::AbstractString
end
Node(ind::Int, S::Type{<:IndexSpace}, desc="") = Node{S, sectortype(S)}(ind, desc)


function hilbertspace(nd::Node{S, Trivial}, sectors::Int, maxdim::Int) where{S}
    dm = min(maxdim, sectors)
    return space(nd)(dm)
end



# trivial node fast constructor.. may be depricated in the future
function Node(s::Int, desc=""; field::Union{Field, Type{<:EuclideanSpace}} = ComplexSpace)
    hilb = field isa Field ? field^1 : field(1)
    return Node(s, typeof(hilb), desc)
end

struct TrivialNode{S<:IndexSpace} <: PhysicalNode{S,Trivial}
    s::Int
    hilbertspace::S
    desc::AbstractString
    function TrivialNode(pos::Int, desc::AbstractString="";
                        dim::Int = 2,
                        field::Union{Field, Type{<:EuclideanSpace}} = ComplexSpace)
        hilb = field isa Field ? field^dim : field(dim)
        return new{typeof(hilb)}(pos, hilb, desc)
    end
end