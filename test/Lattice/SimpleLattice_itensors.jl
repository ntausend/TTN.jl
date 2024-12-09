using TTN, ITensors, ITensorMPS
using Test


@testset "Spin Half Node Lattice" begin
    n_sites = 2
    lat = TTN.Square(n_sites, "SpinHalf")

    indices = TTN.siteinds(lat)
    

    @test TTN.nodetype(lat) == TTN.Node{Index, Int64}

    lat2 = TTN.Square(n_sites, indices)
    @test lat == lat2
    lat3 = TTN.Square(n_sites, "SpinHalf")
    @test !(lat == lat3)
    @test TTN.sectortype(lat) == Int64
    @test TTN.is_physical(lat)


end
