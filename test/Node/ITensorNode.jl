using TTN,ITensors, ITensorMPS
using Test

@testset "Phsical Node Properties, Trivial" begin
    # Hardcore boson node
    nd_sh = TTN.ITensorNode(1,"SpinHalf"; conserve_qns = false)

    @test TTN.position(nd_sh) == 1
    @test TTN.description(nd_sh) == "SpinHalf"
    @test TTN.sectortype(nd_sh) == Int64
    @test TTN.spacetype(nd_sh) == Index

    nd_2 = TTN.nodetype(nd_sh)
    @test nd_2 == TTN.Node{TTN.spacetype(nd_sh), Int64}
  

    for st_name in ["Up", "Dn"]
        @test state(nd_sh, st_name) == state(TTN.index(nd_sh), st_name)
    end
    @test TTN.space(nd_sh) == 2
end

@testset "Phsical Node Properties, ITensorNode, U1" begin
    # Hardcore boson node
    nd_sh = TTN.ITensorNode(1,"SpinHalf"; conserve_qns = true)

    @test TTN.position(nd_sh) == 1
    @test TTN.description(nd_sh) == "SpinHalf"
    @test TTN.sectortype(nd_sh) == Vector{Pair{QN, Int64}}
    @test TTN.spacetype(nd_sh) == Index

    nd_2 = TTN.nodetype(nd_sh)
    @test nd_2 == TTN.Node{Index, Vector{Pair{QN, Int64}}}
    @test !(nd_2 == nd_sh)

    for (st, st_name) in zip([[1,0], [0,1]], ["Up", "Dn"])
        @test TTN.state(nd_sh, st_name) == TTN.state(TTN.index(nd_sh), st_name)
    end
    it_spaces = [QN("Sz", 1) => 1, QN("Sz",-1) => 1]
    @test TTN.space(nd_sh) == it_spaces
end
