module TTN
    using Adapt
    using SparseArrays
    using ITensors
    using ITensorMPS
    using CUDA
    using Distributions: Multinomial
    using KrylovKit
    # using KrylovKit: exponentiate, eigsolve, svdsolve
    using LinearAlgebra
    using Printf
    using HDF5
    # using VectorInterface

    ################## WORKAROUND FOR CURRENT BROKEN ITENSOR FACTORIZE ######################
    #include("factorize_workaround.jl")
    # should be fixed, keep the file in the folder in case we need it.


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

    struct QuantumNumberMissmatch <: Exception end
    Base.showerror(io::IO, ::QuantumNumberMissmatch) = print(io, "Quantum number combination not allowed.")

    struct IndexMissmatchException <: Exception 
        idx::Index
        desc::String
    end
    Base.showerror(io::IO, e::IndexMissmatchException) = print(io, "Index $(e.idx) not fullfill requirements: $(e.desc)") 


    # imports
    import Base: eachindex, size, ==, getindex, setindex, iterate, length, show, copy, eltype
    import ITensors: state, op, space, siteinds, inner
    #import ITensorMPS: expect
    using ITensorMPS: sortmergeterms, determineValType
    using ITensors: terms, which_op, site, params
    #ITensorMPS.sortmergeterms, , ITensorMPS.determineValType
	using ITensors: argument, optimal_contraction_sequence


    # contract_tensor ncon wrapper
    export contract_tensos
    include("./contract_tensors.jl")

    # nodes
    #export TrivialNode, HardCoreBosonNode, SpinHalfNode, Node, ITensorNode
    include("./Node/AbstractNode.jl")
    include("./Node/Node.jl")
    include("./Node/ITensorNode.jl")

    # lattice class
    export Rectangle, Square, Chain
    export linear_ind, coordinate, dimensionality, number_of_sites, coordinates
    include("./Lattice/AbstractLattice.jl")
    include("./Lattice/SimpleLattice.jl")

    # including the Network classes
    export BinaryNetwork, BinaryChainNetwork, BinaryRectangularNetwork
    export lattices, lattice, physical_lattice, number_of_layers, number_of_tensors, dimensions
    export physical_coordinates, eachlayer
    include("./Network/AbstractNetwork.jl")
    include("./Network/BinaryNetwork.jl")
    include("./Network/TernaryNetwork.jl")
    include("./Network/TwelveNetwork.jl")

    

    #=================================================================================#
    # i rather not like to have these kind of functions to be exported...
    #export increase_dim_tree_tensor_network_zeros, increase_dim_tree_tensor_network_randn
    #=================================================================================#

    export TreeTensorNetwork, RandomTreeTensorNetwork, ProductTreeTensorNetwork
    export move_ortho!, adjust_tree_tensor_dimensions, adjust_tree_tensor_dimensions!
    export layer, number_of_layers, network, ortho_center, is_orthogonalized
    export correlation, correlations, entanglement_entropy, inner
    include("./TreeTensorNetwork/TreeTensorNetwork.jl")
    include("./TreeTensorNetwork/algorithms/inner.jl")
    include("./TreeTensorNetwork/algorithms/expect.jl")
    include("./TreeTensorNetwork/algorithms/correlation.jl")

    include("./TreeTensorNetwork/algorithms/entanglement_measures.jl")
    #include("./TreeTensorNetwork/algorithms/observables.jl")

    include("./TPO/AbstractTPO.jl")
    include("./TPO/AbstractProjectedTensorProductOperator.jl")

    # gpu helper functions
    export gpu, cpu
    include("./gpu.jl")
    # MPO class
    export Hamiltonian, ProjMPO, nearest_neighbours
    include("./TPO/ProjMPO/MPO.jl")
    include("./TPO/ProjMPO/ProjectedMatrixProductOperator.jl")
    include("./TPO/ProjMPO/utilsMPO.jl")

    # tensor product operator implementations
    export TPO, ProjTPO
    include("./TPO/ProjTPO/TPO.jl")
    include("./TPO/ProjTPO/ProjectedTensorProductOperator.jl")

    # dmrg/tdvp
    export DefaultExpander, NoExpander
    include("./algorithms/SubspaceExpansion/AbstractSubspaceExpansion.jl")

    export dmrg, tdvp
    include("./algorithms/SweepHandler/AbstractSweepHandler.jl")
    include("./algorithms/SweepHandler/SimpleSweepHandler.jl")
    include("./algorithms/SweepHandler/TDVPSweepHandler.jl")
    include("./algorithms/sweeps.jl")
    include("./algorithms/custom_krylov.jl")

end # module
