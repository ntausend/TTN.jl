struct Node{S, I} <: AbstractNode{S, I}
    s::Int
    desc::AbstractString
end

# trivial node fast constructor.. may be depricated in the future
Node(s::Int, desc="") = Node(s, desc)
function Node(s::Int, desc::AbstractString)
    return Node(s, Int64, desc)
end
Node(ind::Int, ::Type{Int64}, desc="") = Node{Index, Int64}(ind, desc)


# returns an (virtual) node copy of the input node with linear position `s`
# and description `desc`, fogets about possible additional structure if `nd`
# is a PhysicalNode
nodetype(nd::AbstractNode) = nodetype(typeof(nd))
nodetype(::Type{<:AbstractNode{S,I}}) where{S,I} = Node{S,I}
Node(nd::AbstractNode, s::Int, desc::AbstractString="") = nodetype(nd)(s,desc)

#=
function hilbertspace(nd::Node{S, Trivial}, sectors::Int, maxdim::Int) where{S}
    dm = min(maxdim, sectors)
    return space(nd)(dm)
end
=#

struct TrivialNode{S, I} <: PhysicalNode{S,I}
    s::Int
    hilbertspace::S
    desc::AbstractString
end
TrivialNode(pos::Int, desc::AbstractString="";
                    local_dim::Int = 2) = TrivialNode(pos, desc, local_dim)
function TrivialNode(pos::Int, desc::AbstractString, local_dim::Int)
    idx = Index(local_dim, "Site, $desc, n=$pos")
    return TrivialNode{Index, Int64}(pos, idx, desc)
end
function TrivialNode(pos::Int, idx::Index{Int64}, additional_desc = "")
    desc = string(tags(idx)) * string(additional_desc)
    return TrivialNode{Index, Int64}(pos, idx, desc)
end
#=

function TrivialNode(pos::Int, desc::AbstractString, local_dim::Int, backend::TensorKitBackend)
    field = backend.field
    hilb = field isa Field ? field^local_dim : field(local_dim)
    return TrivialNode{typeof(hilb), Trivial}(pos, hilb, desc) 
    #new{typeof(hilb), Trivial}(pos, hilb, desc)
end
=#