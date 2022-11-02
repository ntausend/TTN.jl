struct SpinHalfNode{S<:IndexSpace} <: PhysicalNode{S,Trivial}
    s::Int
    hilbertspace::S
    desc::AbstractString
    function SpinHalfNode(pos::Int, desc::AbstractString="";
                          local_dim::Int = 2,
                          field::Union{Field, Type{<:EuclideanSpace}} = ComplexSpace)
        hilb = field isa Field ? field^local_dim : field(local_dim)
        return new{typeof(hilb)}(pos, hilb, "SH "*desc)
    end
end

state(::SpinHalfNode,::Val{:Up}) = [1, 0]
state(::SpinHalfNode,::Val{:Down}) = [0, 1]
state(::SpinHalfNode,::Val{:Right}) = [1/sqrt(2), 1/sqrt(2)]