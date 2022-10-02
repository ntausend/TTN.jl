struct Node
    s::Int
    hilbertpace::EuclideanSpace
    desc::AbstractString
end

Node(s::Int, hilbertspace::EuclideanSpace) = Node(s, hilbertspace, "")

function Base.show(io::IO, nd::Node)
    print(io, nd.desc)
end

hilbertspace(nd::Node) = nd.hilbertpace