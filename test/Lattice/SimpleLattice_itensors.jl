using TTNKit, ITensors
using Test

@testset "Trivial Simple Lattice, ITensors" begin
    n_sites = 2
    dim  = 2
    backend = TTNKit.ITensorsBackend()
    lat = TTNKit.SimpleLattice((n_sites,n_sites), dim; backend = backend)

    indices = TTNKit.siteinds(lat)

    @test TTNKit.dimensionality(lat) == 2
    @test TTNKit.spacetype(lat)  == Index
    @test TTNKit.sectortype(lat) == Int64
    @test_broken TTNKit.node(lat, 1) == TTNKit.TrivialNode(1,indices[1])
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
    #=
    @test TTNKit.coordinates(lat) == coordinates
    lat2 = TTNKit.Square(n_sites, indices)
    @test_broken lat == lat2
    lat3 = TTNKit.Square(n_sites^2, dim, backend = backend)
    @test !(lat == lat3)
    lat4 = TTNKit.Chain(n_sites^2, indices)
    @test_broken !(lat == lat4)
    =#
end

@testset "Spin Half Node Lattice, ITensors" begin
    n_sites = 2
    ndtype = TTNKit.ITensorNode
    lat = TTNKit.Square(n_sites, ndtype, "SpinHalf")

    indices = TTNKit.siteinds(lat)
    

    @test_broken TTNKit.nodetype(lat) == TTNKit.ITensorNode

    #=
    lat2 = TTNKit.Square(n_sites, indices)
    @test lat == lat2
    lat3 = TTNKit.Square(n_sites^2, indices)
    @test !(lat == lat3)
    @test TTNKit.sectortype(lat) == Int64
    @test TTNKit.is_physical(lat)
    =#


end