using Test

println("Running Tests....")

@testset "TTNKit.jl" begin
    @testset "$filename"  for filename in [
        "Node.jl",
        "HardCoreBosonNode.jl",
        "SoftCoreBosonNode.jl",
        "Lattice.jl",
        "Network.jl",
        "TreeTensorNetwork.jl",
        "inner.jl",
        "expect.jl",
        "correlation.jl"
    ]
        println("Running $filename")
        include(filename)
    end
end