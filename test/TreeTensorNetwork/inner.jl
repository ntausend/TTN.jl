using TTNKit
using Test


@testset "Inner product" begin
    n_layers = 2
    net = TTNKit.BinaryChainNetwork(2)
    
    ttn = TTNKit.RandomTreeTensorNetwork(net, orthogonalize = true)
    
    @test TTNKit.inner(ttn, ttn) ≈ 1
    
    ttnc = copy(ttn)
    TTNKit.move_ortho!(ttn, (1,1))
    @test TTNKit.inner(ttn, ttnc) ≈ 1
    

    maxdim = 30
    ttn = TTNKit.RandomTreeTensorNetwork(net; maxdim = maxdim, orthogonalize = true, normalize = false)
    @test TTNKit.inner(ttn,ttn) ≈ TTNKit.norm(ttn)^2

    TTNKit.normalize!(ttn)
    @test TTNKit.norm(ttn) ≈ 1
end