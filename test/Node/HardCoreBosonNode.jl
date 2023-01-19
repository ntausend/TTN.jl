using TTNKit, TensorKit
using Test


elT = Float64

@testset "Phsical Node Properties, Hard Core Boson, Trivial" begin
    # Hardcore boson node
    nd_hcb = TTNKit.HardCoreBosonNode(1,"1"; conserve_qns = false)

    @test TTNKit.position(nd_hcb) == 1
    @test TTNKit.description(nd_hcb) == "HCB 1"
    @test sectortype(nd_hcb) == Trivial
    @test spacetype(nd_hcb) <: ComplexSpace

    nd_2 = TTNKit.nodetype(nd_hcb)
    @test nd_2 == TTNKit.Node{spacetype(nd_hcb), Trivial}
    @test !(nd_2 == nd_hcb)

    for (st, st_name) in zip([[1,0], [0,1]], ["Emp", "Occ"])
        @test TTNKit.state(nd_hcb, Val(Symbol(st_name))) == st
        st_o = TTNKit.state(nd_hcb, st_name, elT)
        @test st_o == TensorMap(elT.(st), ℂ^2 ← ℂ^1)
    end
    @test TTNKit.space(nd_hcb) == 2
end

@testset "Phsical Node Properties, Hard Core Boson, U1" begin
    # Hardcore boson node
    nd_hcb = TTNKit.HardCoreBosonNode(1,"1"; conserve_qns = true)

    @test TTNKit.position(nd_hcb) == 1
    @test TTNKit.description(nd_hcb) == "HCB 1"
    @test sectortype(nd_hcb) == U1Irrep
    @test spacetype(nd_hcb) <: GradedSpace

    nd_2 = TTNKit.nodetype(nd_hcb)
    @test nd_2 == TTNKit.Node{spacetype(nd_hcb), U1Irrep}
    @test !(nd_2 == nd_hcb)

    for (st, st_name) in zip([[1,0], [0,1]], ["Emp", "Occ"])
        @test TTNKit.state(nd_hcb, Val(Symbol(st_name))) == st
    end
    @test TTNKit.space(nd_hcb) == [0 => 1, 1 => 1]


    st_o = TTNKit.state(nd_hcb, "Emp", elT)
    @test st_o == TensorMap(ones, elT, U1Space(0=> 1, 1=>1)← U1Space(0 => 1))

    st_o = TTNKit.state(nd_hcb, "Occ", elT)
    @test st_o == TensorMap(ones, elT, U1Space(0=> 1, 1=>1)← U1Space(1 => 1))
end

@testset "Phsical Node Properties, Hard Core Boson, Z2" begin
    # Hardcore boson node
    nd_hcb = TTNKit.HardCoreBosonNode(1,"1"; conserve_qns = false, conserve_parity = true)

    @test TTNKit.position(nd_hcb) == 1
    @test TTNKit.description(nd_hcb) == "HCB 1"
    @test sectortype(nd_hcb) == Z2Irrep
    @test spacetype(nd_hcb) <: GradedSpace

    nd_2 = TTNKit.nodetype(nd_hcb)
    @test nd_2 == TTNKit.Node{spacetype(nd_hcb), Z2Irrep}
    @test !(nd_2 == nd_hcb)

    for (st, st_name) in zip([[1,0], [0,1]], ["Emp", "Occ"])
        @test TTNKit.state(nd_hcb, Val(Symbol(st_name))) == st
    end
    @test TTNKit.space(nd_hcb) == [0 => 1, 1 => 1]

    st_o = TTNKit.state(nd_hcb, "Emp", elT)
    @test st_o == TensorMap(ones, elT, Z2Space(0=> 1, 1=>1)← Z2Space(0 => 1))

    st_o = TTNKit.state(nd_hcb, "Occ", elT)
    @test st_o == TensorMap(ones, elT, Z2Space(0=> 1, 1=>1)← Z2Space(1 => 1))
end