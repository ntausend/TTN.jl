using TTNKit, TensorKit, ITensors
using Test


@testset "Basic Lattice Properties, TensorKit" begin
    n_sites = 8
    backend = TTNKit.TensorKitBackend()
    lat = TTNKit.CreateChain(n_sites; backend = backend)

    @test TTNKit.dimensionality(lat) == 1
    @test spacetype(lat)  == ComplexSpace
    @test sectortype(lat) == Trivial
    @test TTNKit.node(lat, 1) == TTNKit.Node(1,"1"; backend = backend)
    @test TTNKit.number_of_sites(lat) == n_sites
    @test_throws TTNKit.NotImplemented TTNKit.linear_ind(lat,(1,))
    @test_throws TTNKit.NotImplemented TTNKit.coordinate(lat,1)
    @test !TTNKit.is_physical(lat)
    @test TTNKit.nodetype(lat) == TTNKit.Node{ComplexSpace, Trivial}

    for (jj,nd) in enumerate(lat)
        @test nd == TTNKit.Node(jj,"$jj"; backend = backend)
    end
    @test length(lat) == n_sites
    @test lat[1] == TTNKit.Node(1,"1"; backend = backend)
    @test eachindex(lat) == 1:8
    @test_throws TTNKit.DimensionsException TTNKit.CreateChain(n_sites - 1; backend = backend)
    @test_throws TTNKit.siteinds(lat)
end

@testset "Basic Lattice Properties, ITensors" begin
    n_sites = 8
    backend = TTNKit.ITensorsBackend()

    lat = TTNKit.CreateChain(n_sites; backend = backend)

    @test TTNKit.dimensionality(lat) == 1
    @test spacetype(lat)  == Index
    @test sectortype(lat) == Int64
    @test TTNKit.node(lat, 1) == TTNKit.Node(1,"1"; backend = backend)
    @test TTNKit.number_of_sites(lat) == n_sites
    @test_throws TTNKit.NotImplemented TTNKit.linear_ind(lat,(1,))
    @test_throws TTNKit.NotImplemented TTNKit.coordinate(lat,1)
    @test !TTNKit.is_physical(lat)
    @test TTNKit.nodetype(lat) == TTNKit.Node{Index, Int64}
    for (jj,nd) in enumerate(lat)
        @test nd == TTNKit.Node(jj,"$jj"; backend = backend)
    end
    @test length(lat) == n_sites
    @test lat[1] == TTNKit.Node(1,"1"; backend  = backend)
    @test eachindex(lat) == 1:8
    @test_throws TTNKit.DimensionsException TTNKit.CreateChain(n_sites - 1; backend = backend)
    @test_throws TTNKit.siteinds(lat)
end