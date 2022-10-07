@testset "Inner product" begin
    n_layers = 2
    ndtyp = TTNKit.TrivialNode
    net = BinaryChainNetwork(2, ndtyp)
    
    ttn = RandomTreeTensorNetwork(net, orthogonalize = true)
    
    @test TTNKit.inner(ttn, ttn) ≈ 1
    
    ttnc = copy(ttn)
    TTNKit.move_ortho!(ttn, (1,1))
    @test TTNKit.inner(ttn, ttnc) ≈ 1
    

    maxdim = 30
    ttn = RandomTreeTensorNetwork(net; maxdim = maxdim, orthogonalize = true, normalize = false)
    @test TTNKit.inner(ttn,ttn) ≈ TTNKit.norm(ttn)^2

    normalize!(ttn)
    @tensor TTNKit.norm(ttn) ≈ 1

end