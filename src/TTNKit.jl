module TTNKit
    using SparseArrays
    using TensorKit
    using ITensors
    using ITensorGPU
    using Distributions: Multinomial
    using Parameters: @with_kw
    using MPSKit: MPOHamiltonian, DenseMPO, _embedders, SparseMPO, PeriodicArray
    using MPSKitModels: LocalOperator, @mpoham
    using KrylovKit
    using LinearAlgebra
    using Printf


    


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
    import TensorKit: sectortype, spacetype
    import ITensors: state, op, space, siteinds
    import ITensors: expect
    using ITensors: terms, sortmergeterms, which_op, site, params, determineValType
	using ITensors: argument, optimal_contraction_sequence
    using ITensors:  dim as dim_it
    using TensorKit: dim as dim_tk
    using ITensors:  dims as dims_it
    using TensorKit: dims as dims_tk

    dim(ind::I) where{I} = I <: Index ? dim_it(ind) : dim_tk(ind) 

    # fixing missing support for qn-sparse qr decomposition of ITensors, should
    # be included in the future.. see pullrequest:
    # https://github.com/ITensor/ITensors.jl/pull/1009
    # This is also my code from 
    # https://github.com/ntausend/variance_iTensor
    # in slightly modified version of Jan Reimers
    # just use the factorize for the moment... dont want to get nasty warnings
    #include("./qn_qr_it/qr.jl")

    include("./backends/backends.jl")

    # contract_tensor ncon wrapper
    include("./contract_tensors.jl")

    # nodes
    export TrivialNode, HardCoreBosonNode, SpinHalfNode, Node, ITensorNode
    include("./Node/AbstractNode.jl")
    include("./Node/Node.jl")
    include("./Node/ITensorNode.jl")
    include("./Node/HardCoreBosonNode.jl")
    include("./Node/SoftCoreBosonNode.jl")
    include("./Node/SpinHalfNode.jl")

    # lattice class
    export Rectangle, Square
    include("./Lattice/AbstractLattice.jl")
    include("./Lattice/SimpleLattice.jl")

    # including the Network classes
    export BinaryNetwork, BinaryChainNetwork, BinaryRectangularNetwork
    include("./Network/AbstractNetwork.jl")
    include("./Network/BinaryNetwork.jl")

    

    #=================================================================================#
    # i rather not like to have these kind of functions to be exported...
    export increase_dim_tree_tensor_network_zeros, increase_dim_tree_tensor_network_randn
    #=================================================================================#

    export TreeTensorNetwork, RandomTreeTensorNetwork, ProductTreeTensorNetwork
    export move_ortho!, adjust_tree_tensor_dimensions, adjust_tree_tensor_dimensions!
    include("./TreeTensorNetwork/TreeTensorNetwork.jl")
    include("./TreeTensorNetwork/algorithms/inner.jl")
    include("./TreeTensorNetwork/algorithms/expect.jl")

    #=================================================================================#
    # Does this function still exists?
    #=================================================================================#
    #export transverseIsingHamiltonian
    #=================================================================================#

    include("./TPO/AbstractProjectedTensorProductOperator.jl")
    include("./TPO/AbstractTPO.jl")
    # MPO class
    include("./TPO/ProjMPO/MPO.jl")
    # abstract TPO
    include("./TPO/ProjMPO/ProjectedMatrixProductOperator.jl")
    include("./TPO/ProjMPO/utilsMPO.jl")

    # tensor product operator implementations
    include("./TPO/ProjTPO/ProjectedTensorProductOperator.jl")

    # model implementations
    include("./Models/TransverseFieldIsing.jl")
    include("./Models/TrivialModel.jl")


    # dmrg/tdvp
    include("./algorithms/SubspaceExpansion/AbstractSubspaceExpansion.jl")
    include("./algorithms/SweepHandler/AbstractSweepHandler.jl")
    include("./algorithms/SweepHandler/SimpleSweepHandler.jl")
    include("./algorithms/SweepHandler/TDVPSweepHandler.jl")
    include("./algorithms/sweeps.jl")


    #=

    
    export TreeTensorNetwork, RandomTreeTensorNetwork, ProductTreeTensorNetwork
    

    # load the definition of special operator types for dispatching measuring functions
    include("./TPO/AbstractTensorDefinitions.jl")

    include("./TreeTensorNetwork/algorithms/correlation.jl")
   =# 

    #============================= TENSOR PRODUCT OPERATORS =========================#


    #=


    # TPO TODO:Tests

    include("./TPO/TPOSum/Interactions.jl")
    include("./TPO/TPOSum/TPOSum.jl")
    =#

end # module
