using TTNKit,ITensors
using Test

@testset "Phsical Node Properties, ITensorNode, Trivial" begin
    # Hardcore boson node
    nd_sh = TTNKit.ITensorNode(1,"SpinHalf"; conserve_qns = false)

    @test TTNKit.position(nd_sh) == 1
    @test TTNKit.description(nd_sh) == "SpinHalf 1"
    @test TTNKit.sectortype(nd_sh) == Int64
    @test TTNKit.spacetype(nd_sh) == Index

    nd_2 = TTNKit.nodetype(nd_sh)
    @test nd_2 == TTNKit.Node{TTNKit.spacetype(nd_sh), Int64}
  

    for st_name in ["Up", "Dn"]
        @test state(nd_sh, st_name) == state(TTNKit.index(nd_sh), st_name)
    end
    @test TTNKit.space(nd_sh) == 2
end

@testset "Phsical Node Properties, ITensorNode, U1" begin
    # Hardcore boson node
    nd_sh = TTNKit.ITensorNode(1,"SpinHalf"; conserve_qns = true)

    @test TTNKit.position(nd_sh) == 1
    @test TTNKit.description(nd_sh) == "SpinHalf 1"
    @test sectortype(nd_sh) == Vector{Pair{QN, Int64}}
    @test spacetype(nd_sh) == Index

    nd_2 = TTNKit.nodetype(nd_sh)
    @test nd_2 == TTNKit.Node{Index, Vector{Pair{QN, Int64}}}
    @test !(nd_2 == nd_sh)

    for (st, st_name) in zip([[1,0], [0,1]], ["Up", "Dn"])
        @test state(nd_sh, st_name) == state(TTNKit.index(nd_sh), st_name)
    end
    it_spaces = [QN("Sz", 1) => 1, QN("Sz",-1) => 1]
    @test TTNKit.space(nd_sh) == it_spaces
end