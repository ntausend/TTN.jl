using TTNKit, TensorKit
using Test


elT = Float64

@testset "Phsical Node Properties, SpinHalfNode, Trivial" begin
    # Hardcore boson node
    nd_sh = TTNKit.SpinHalfNode(1,"1"; conserve_qns = false)

    @test TTNKit.position(nd_sh) == 1
    @test TTNKit.description(nd_sh) == "SH 1"
    @test sectortype(nd_sh) == Trivial
    @test spacetype(nd_sh) <: ComplexSpace

    nd_2 = TTNKit.nodetype(nd_sh)
    @test nd_2 == TTNKit.Node{spacetype(nd_sh), Trivial}
    @test !(nd_2 == nd_sh)

    for (st, st_name) in zip([[1,0], [0,1], [1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]], 
                              ["Up", "Down", "Right",            "Left"])
        @test TTNKit.state(nd_sh, Val(Symbol(st_name))) == st
        st_o = TTNKit.state(nd_sh, st_name, elT)
        @test st_o == TensorMap(elT.(st), ℂ^2 ← ℂ^1)
    end
    @test TTNKit.space(nd_sh) == 2
end

@testset "Phsical Node Properties, SpinHalfNode, U1" begin
    # Hardcore boson node
    nd_sh = TTNKit.SpinHalfNode(1,"1"; conserve_qns = true)

    @test TTNKit.position(nd_sh) == 1
    @test TTNKit.description(nd_sh) == "SH 1"
    @test sectortype(nd_sh) == U1Irrep
    @test spacetype(nd_sh) <: GradedSpace

    nd_2 = TTNKit.nodetype(nd_sh)
    @test nd_2 == TTNKit.Node{spacetype(nd_sh), U1Irrep}
    @test !(nd_2 == nd_sh)

    for (st, st_name) in zip([[1,0], [0,1], [1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]], 
                              ["Up", "Down", "Right",            "Left"])
        @test TTNKit.state(nd_sh, Val(Symbol(st_name))) == st
    end
    @test TTNKit.space(nd_sh) == [0 => 1, 1 => 1]


    st_o = TTNKit.state(nd_sh, "Up", elT)
    @test st_o == TensorMap(ones, elT, U1Space(0=> 1, 1=>1)← U1Space(0 => 1))

    st_o = TTNKit.state(nd_sh, "Down", elT)
    @test st_o == TensorMap(ones, elT, U1Space(0=> 1, 1=>1)← U1Space(1 => 1))
    # left right are not conserving the QN numbers so not allowed here!
end

@testset "Phsical Node Properties, SpinHalfNode, Z2" begin
    # Hardcore boson node
    nd_sh = TTNKit.SpinHalfNode(1,"1"; conserve_qns = false, conserve_parity = true)

    @test TTNKit.position(nd_sh) == 1
    @test TTNKit.description(nd_sh) == "SH 1"
    @test sectortype(nd_sh) == Z2Irrep
    @test spacetype(nd_sh) <: GradedSpace

    nd_2 = TTNKit.nodetype(nd_sh)
    @test nd_2 == TTNKit.Node{spacetype(nd_sh), Z2Irrep}
    @test !(nd_2 == nd_sh)

    for (st, st_name) in zip([[1,0], [0,1], [1/sqrt(2), 1/sqrt(2)], [1/sqrt(2), -1/sqrt(2)]], 
                              ["Up", "Down", "Right",            "Left"])
        @test TTNKit.state(nd_sh, Val(Symbol(st_name))) == st
    end
    @test TTNKit.space(nd_sh) == [0 => 1, 1 => 1]

    st_o = TTNKit.state(nd_sh, "Up", elT)
    @test st_o == TensorMap(ones, elT, Z2Space(0=> 1, 1=>1)← Z2Space(0 => 1))

    st_o = TTNKit.state(nd_sh, "Down", elT)
    @test st_o == TensorMap(ones, elT, Z2Space(0=> 1, 1=>1)← Z2Space(1 => 1))
end