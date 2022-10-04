struct Node
    s::Int
    hilbertpace::EuclideanSpace
    desc::AbstractString
end

Node(s::Int, hilbertspace::EuclideanSpace) = Node(s, hilbertspace, "")

function Base.show(io::IO, nd::Node, show_hilbertspace = true)
    s = "Node ($(nd.desc)), Number: $(nd.s)"
    show_hilbertspace && (s *= " Hilbertpace: $(nd.hilbertpace)")
    print(io, s)
end

position(nd::Node) = nd.s
description(nd::Node) = nd.desc

hilbertspace(nd::Node) = nd.hilbertpace