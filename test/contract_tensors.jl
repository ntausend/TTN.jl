using TTN
using Test

@testset "Contraction TTN Network" begin
    n_layers = 2
    net = TTN.BinaryChainNetwork(n_layers)
    ttn = TTN.RandomTreeTensorNetwork(net)
    tensors = vcat(ttn.data...)

    inds = map(p -> TTN.internal_index_of_legs(net, p), TTN.NodeIterator(net))
    unique_inds, cntr_tn = TTN.contract_tensors(tensors, inds)

    res_normal = (tensors[1]*tensors[2])*tensors[3]

    @test res_normal ≈ cntr_tn

end
