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

@testset "Phsical Node Properties, Hard Core Boson" begin
    # Hardcore boson node
    nd_hcb = TTNKit.HardCoreBosonNode(1,"1"; conserve_qns = true)

    @test TTNKit.position(nd_hcb) == 1
    @test TTNKit.description(nd_hcb) == "HCB 1"
    @test sectortype(nd_hcb) == U1Irrep
    @test spacetype(nd_hcb) <: GradedSpace

    nd_2 = TTNKit.nodetype(nd_hcb)
    @test nd_2 == Node{spacetype(nd_hcb), U1Irrep}
    @test !(nd_2 == nd_hcb)

    d_st = TTNKit.state_dict(typeof(nd_hcb))
    @test d_st["Emp"] == [1, 0]
    @test d_st["Occ"] == [0, 1]

    d_chrg = TTNKit.charge_dict(typeof(nd_hcb))
    @test d_chrg["Emp"] == U1Irrep(0)
    @test d_chrg["Occ"] == U1Irrep(1)

    st_o = TTNKit.state(nd_hcb, "Emp", elT = Float64)
    @test st_o == TensorMap(ones, Float64, U1Space(0=> 1, 1=>1)← U1Space(0 => 1))
end