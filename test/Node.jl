using TTNKit, TensorKit
using Test


@testset "Basic Node Properties" begin
    
    # basic node
    nd_1 = Node(1,"1")
    @test TTNKit.position(nd_1) == 1
    @test TTNKit.description(nd_1) == "1"
    @test spacetype(nd_1)  == ComplexSpace
    @test sectortype(nd_1) == Trivial

    nd_t = TTNKit.nodetype(nd_1)
    @test nd_t == Node{ComplexSpace, Trivial}
    nd_2 = Node(nd_1, 2,"2")
    @test TTNKit.position(nd_2) == 2
    @test TTNKit.description(nd_2) == "2"
    @test spacetype(nd_2)  == ComplexSpace
    @test sectortype(nd_2) == Trivial
    
    local_dim = 2
    @test TTNKit.hilbertspace(nd_1, local_dim) == ℂ^local_dim
end