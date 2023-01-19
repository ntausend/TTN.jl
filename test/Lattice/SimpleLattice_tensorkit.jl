using TTNKit, TensorKit
using Test

@testset "Trivial Simple Lattice, TensorKit" begin
    n_sites = 2
    dim  = 2
    lat = TTNKit.SimpleLattice((n_sites,n_sites), dim)

    @test TTNKit.dimensionality(lat) == 2
    @test spacetype(lat)  == ComplexSpace
    @test sectortype(lat) == Trivial
    @test TTNKit.node(lat, 1) == TTNKit.TrivialNode(1,"1 1")
    @test TTNKit.number_of_sites(lat) == n_sites^2
    @test TTNKit.linear_ind(lat,(2,1)) == 2
    @test TTNKit.linear_ind(lat,(1,2)) == n_sites + 1
    @test TTNKit.coordinate(lat,2) == (2,1)
    @test TTNKit.coordinate(lat, n_sites + 1) == (1,2)
    
    coordinates = [
        (1,1)
        (2,1)
        (1,2)
        (2,2)
    ]
    @test TTNKit.coordinates(lat) == coordinates
    lat2 = TTNKit.Square(n_sites,dim)
    @test lat == lat2
    lat3 = TTNKit.Square(n_sites^2, dim)
    @test !(lat == lat3)
    lat4 = TTNKit.Chain(n_sites,dim)
    @test !(lat == lat4)
end

@testset "Hardcore Boson Node Lattice, TensorKit" begin
    n_sites = 2
    ndtype = TTNKit.HardCoreBosonNode
    lat = TTNKit.Square(n_sites, ndtype)
    # one may need to return the correct node type in case of physcial nodes??
    # this test if false
    @test_broken TTNKit.nodetype(lat) <: TTNKit.HardCoreBosonNode

    lat2 = TTNKit.Square(n_sites, ndtype)
    @test lat == lat2
    lat3 = TTNKit.Square(n_sites^2, ndtype)
    @test !(lat == lat3)
    @test sectortype(lat) == U1Irrep
    @test TTNKit.is_physical(lat)
end