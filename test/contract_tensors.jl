using TensorKit, TTNKit
using Test

@testset "Contraction TTN Network"

    n_layers = 2
    net = TTNKit.BinaryChainNetwork(n_layers)
    ttn = TTNKit.RandomTreeTensorNetwork(net)
    tensors = vcat(ttn.data...)

    inds = map(p -> TTNKit.internal_index_of_legs(net, p), TTNKit.NodeIterator(net))
    unique_inds, cntr_tn = contract_tensors(tensors, inds)


    @tensor res_normal[-1,-2,-3,-4,-5] := tensors[1][-1,-2, 1] * tensors[2][-3,-4,2]*tensors[3][1,2,-5]

    @test res_normal ≈ cntr_tn

end