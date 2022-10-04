module TTNKit
    using SparseArrays
    using TensorKit

    # lattice class

    export AbstractLattice, BinaryChain, BinaryRectangle, BinarySquare
    include("./Lattice/Lattice.jl")
    include("./Lattice/BinaryLattice.jl")

    # including the Network classes

    
    export AbstractNetwork, BinaryNetwork, BinaryChainNetwork, BinaryRectangularNetwork
    include("./Network/AbstractNetwork.jl")
    include("./Network/BinaryNetwork.jl")

    
    export TreeTensorNetwork
    include("./TreeTensorNetwork/TreeTensorNetwork.jl")
end # module
