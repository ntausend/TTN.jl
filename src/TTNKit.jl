module TTNKit
    using SparseArrays
    using TensorKit
    using Distributions: Multinomial

    struct NotImplemented <: Exception
        fn::Symbol
        type_name
    end
    Base.showerror(io::IO, e::NotImplemented) = print(io, e.fn, " not implemented for this type: ", e.type_name)

    struct DimensionsException{D} <: Exception
        dims::NTuple{D,Int}
    end
    function Base.showerror(io::IO, e::DimensionsException{D}) where D
        n_sites = prod(e.dims)
        s_err = "Number of Sites $(n_sites) is not compatible with a binary network of dimension $D"
        s_err *= " with n layers requireing number_of_sites = 2^(n_$(D))"
        print(io, s_err)
    end

    struct NotSupportedException <: Exception
        msg::AbstractString
    end
    Base.showerror(io::IO, e::NotSupportedException) = print(io, "Functionality is not supported: "*e.msg)


    # imports
    import Base: eachindex, size, ==, getindex, setindex, iterate, length, show, copy, eltype
    import TensorKit: sectortype, spacetype


    # nodes
    export TrivialNode, HardCoreBosonNode, Node
    include("./Node/AbstractNode.jl")
    include("./Node/Node.jl")
    include("./Node/HardCoreBosonNode.jl")
    include("./Node/SoftCoreBosonNode.jl")

    # lattice class
    export AbstractLattice, Chain, Rectangle, Square
    include("./Lattice/AbstractLattice.jl")
    include("./Lattice/SimpleLattice.jl")

    # including the Network classes

    
    export BinaryNetwork, BinaryChainNetwork, BinaryRectangularNetwork
    include("./Network/AbstractNetwork.jl")
    include("./Network/BinaryNetwork.jl")

    
    export TreeTensorNetwork, RandomTreeTensorNetwork, ProductTreeTensorNetwork
    include("./TreeTensorNetwork/TreeTensorNetwork.jl")
    
    include("./TreeTensorNetwork/algorithms/inner.jl")

    # load the definition of special operator types for dispatching measuring functions
    include("./TPO/AbstractTensorDefinitions.jl")
    include("./TreeTensorNetwork/algorithms/expect.jl")

    include("./TreeTensorNetwork/algorithms/correlation.jl")
    

    #=


    # TPO TODO:Tests
    include("./TPO/AbstractTPO.jl")
    include("./TPO/ProjTPO.jl")

    include("./TPO/TPOSum/Interactions.jl")
    include("./TPO/TPOSum/TPOSum.jl")
    =#

end # module
