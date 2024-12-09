using TTN
using Test


@testset "Inner product" begin
    n_layers = 2
    net = TTN.BinaryChainNetwork(2)
    
    ttn = TTN.RandomTreeTensorNetwork(net, orthogonalize = true)
    
    @test TTN.inner(ttn, ttn) ≈ 1
    
    ttnc = copy(ttn)
    TTN.move_ortho!(ttn, (1,1))
    @test TTN.inner(ttn, ttnc) ≈ 1
    

    maxdim = 30
    ttn = TTN.RandomTreeTensorNetwork(net; maxdim = maxdim, orthogonalize = true, normalize = false)
    @test TTN.inner(ttn,ttn) ≈ TTN.norm(ttn)^2

    TTN.normalize!(ttn)
    @test TTN.norm(ttn) ≈ 1
end
