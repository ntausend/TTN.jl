using TTNKit, TensorKit
using Test

@testset "General Tree Tensor Network" begin
    n_layers = 2
    ndtyp = TTNKit.TrivialNode
    net = BinaryChainNetwork(2, ndtyp)
    
    ttn = RandomTreeTensorNetwork(net, orthogonalize = false)
    
    @test TTNKit.number_of_layers(ttn) == n_layers
    @test length(TTNKit.layer(ttn,1))  == 2
    @test TTNKit.network(ttn) == net
    @test TTNKit.ortho_center(ttn) == (-1,-1)
    ttn = RandomTreeTensorNetwork(net, orthogonalize = true)
    @test TTNKit.ortho_center(ttn) == (n_layers,1)
    @test ttn[1,1] == ttn[(1,1)]
    
    n_ten = TensorMap(randn, ℂ^2 ⊗ ℂ^2 ← ℂ^1)
    ttn[1,1] = n_ten
    @test ttn[1,1] == n_ten
    ttn = RandomTreeTensorNetwork(net, orthogonalize = true)

    TTNKit.move_down!(ttn,1)
    @test TTNKit.ortho_center(ttn) == (n_layers-1,1)
    TTNKit.move_up!(ttn)
    @test TTNKit.ortho_center(ttn) == (n_layers,1)
    TTNKit.move_ortho!(ttn, (1,1))
    @test TTNKit.ortho_center(ttn) == (1,1)
    
    is_normal, res = TTNKit.check_normality(ttn)
    @test is_normal
    @test res ≈ 1


    elT = Float64
    ttn = RandomTreeTensorNetwork(net; elT = elT)
    @test eltype(ttn) == elT

    elT = ComplexF64
    ttn = RandomTreeTensorNetwork(net; elT = elT)
    @test eltype(ttn) == elT
end