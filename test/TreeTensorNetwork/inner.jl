using TensorKit, TTNKit
using Test

backend = ("TensorKit" => TTNKit.TensorKitBackend(), "ITensors" => TTNKit.ITensorsBackend())

for (ba_name, ba_ttnk) in backend
    @testset "Inner product, $(ba_name)" begin
        n_layers = 2
        ndtyp = TTNKit.TrivialNode
        net = TTNKit.BinaryChainNetwork(2, ndtyp; backend = ba_ttnk)
        
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
end