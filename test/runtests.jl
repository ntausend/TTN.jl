using Test

println("Running Tests....")

@testset "TTNKit.jl" begin
    @testset "$filename"  for filename in [
        "Node.jl",
        "Lattice.jl",
        "Network.jl",
        "TreeTensorNetwork.jl",
        "inner.jl",
        "expect.jl"
    ]
        println("Running $filename")
        include(filename)
    end
end