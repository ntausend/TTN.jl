module TTNKit
    using SparseArrays
    using TensorKit

    # nodes
    export TrivialNode, HardCoreBosonNode, Node
    include("./Node/AbstractNode.jl")
    include("./Node/Node.jl")
    include("./Node/HardCoreBosonNode.jl")

    # lattice class
    export AbstractLattice, BinaryChain, BinaryRectangle, BinarySquare
    include("./Lattice/AbstractLattice.jl")
    include("./Lattice/BinaryLattice.jl")

    # including the Network classes

    
    export BinaryNetwork, BinaryChainNetwork, BinaryRectangularNetwork
    include("./Network/AbstractNetwork.jl")
    include("./Network/BinaryNetwork.jl")

    
    export TreeTensorNetwork, RandomTreeTensorNetwork, ProductTreeTensorNetwork
    include("./TreeTensorNetwork/TreeTensorNetwork.jl")
end # module
