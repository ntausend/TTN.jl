using Test

println("Running Tests....")

@testset "TTNKit.jl" begin
    @testset "$filename"  for filename in [
        #"Node/Node.jl",
        "Node/ITensorNode.jl",
        #"Lattice/GenericLattice.jl",
        #"Lattice/SimpleLattice_tensorkit.jl",
        "Lattice/SimpleLattice_itensors.jl",
        "Network/Network.jl",
        #"Network/Network_tensorkit.jl",
        "Network/Network_itensors.jl",
        "TreeTensorNetwork/TreeTensorNetwork.jl",
        "TreeTensorNetwork/inner.jl",
        #"TreeTensorNetwork/expect_tensorkit.jl",
        "TreeTensorNetwork/expect_itensors.jl",
        "correlation.jl",
        "contract_tensors.jl",
        "environments.jl",
        "sweep_protocols.jl",
        ##"SubspaceExpansion.jl",
        ##"DMRG.jl",
    ]
        println("Running $filename")
        include(filename)
    end
end
