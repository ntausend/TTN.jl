using Test

println("Running Tests....")

@testset "TTNKit.jl" begin
    @testset "$filename"  for filename in [
        "Node.jl",
        "HardCoreBosonNode.jl",
        "SoftCoreBosonNode.jl",
        "SpinHalfNode.jl",
        "Lattice.jl",
        "Network.jl",
        #"sweep_handlers.jl",
        "TreeTensorNetwork.jl",
        "inner.jl",
        "expect.jl",
        "correlation.jl",
        "contract_tensors.jl",
        "environments.jl"
    ]
        println("Running $filename")
        include(filename)
    end
end
