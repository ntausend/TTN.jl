struct Node
    s::Int64
    desc::String
end

Node(s::Int) = Node(s, "")

function Base.show(io::IO, nd::Node)
    print(io, nd.desc)
end

#const Lattice = Vector{Node}