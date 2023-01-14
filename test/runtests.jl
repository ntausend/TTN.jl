using Test

println("Running Tests....")

@testset "TTNKit.jl" begin
    @testset "$filename"  for filename in [
        "Node/Node.jl",
        "Node/HardCoreBosonNode.jl",
        "Node/SoftCoreBosonNode.jl",
        "Node/SpinHalfNode.jl",
        "Node/ITensorNode.jl",
        "Lattice/GenericLattice.jl",
        "Lattice/SimpleLattice_tensorkit.jl",
        "Lattice/SimpleLattice_itensors.jl",
        "Network/Network.jl",
        "Network/Network_tensorkit.jl",
        "Network/Network_itensors.jl",
        "TreeTensorNetwork.jl",
        #"inner.jl",
        #"expect.jl",
        #"correlation.jl",
        #"contract_tensors.jl",
        #"environments.jl",
        #"sweep_handlers.jl",
    ]
        println("Running $filename")
        include(filename)
    end
end
