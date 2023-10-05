using TTNKit
using Test

@testset "Contraction TTN Network" begin
    n_layers = 2
    net = TTNKit.BinaryChainNetwork(n_layers)
    ttn = TTNKit.RandomTreeTensorNetwork(net)
    tensors = vcat(ttn.data...)

    inds = map(p -> TTNKit.internal_index_of_legs(net, p), TTNKit.NodeIterator(net))
    unique_inds, cntr_tn = TTNKit.contract_tensors(tensors, inds)

    res_normal = (tensors[1]*tensors[2])*tensors[3]

    @test res_normal ≈ cntr_tn

end