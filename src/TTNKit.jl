module TTNKit
    using SparseArrays
    using TensorKit

    # lattice class

    export AbstractLattice, BinaryChain, CreateBinaryChain
    include("./Lattice/Lattice.jl")
    include("./Lattice/BinaryChain.jl")

    # including the Network classes

    export AbstractNetwork, Network, CreateBinaryNetwork
    export OneDimensionalBinaryNetwork
    #export dimensionality, n_tensors, n_layers, n_tensors
    #export adjacencyMatrix, bonddim, NetworkBinaryOneDim
    #export parentNode, childNodes, connectingPath
    include("./Network/AbstractNetwork.jl")
    include("./Network/OneDimensionalBinaryNetwork.jl")


    export TreeTensorNetwork
    include("./TreeTensorNetwork/TreeTensorNetwork.jl")

end # module
