using TensorKit, TTNKit
using Test

@testset "Contraction TTN Network, TensorKit" begin

    n_layers = 2
    backend = TTNKit.TensorKitBackend()
    net = TTNKit.BinaryChainNetwork(n_layers; backend = backend)
    ttn = TTNKit.RandomTreeTensorNetwork(net)
    tensors = vcat(ttn.data...)

    inds = map(p -> TTNKit.internal_index_of_legs(net, p), TTNKit.NodeIterator(net))
    unique_inds, cntr_tn = TTNKit.contract_tensors(tensors, inds)


    @tensor res_normal[-1,-2,-3,-4,-5] := tensors[1][-1,-2, 1] * tensors[2][-3,-4,2]*tensors[3][1,2,-5]

    @test res_normal ≈ cntr_tn

end
@testset "Contraction TTN Network, ITensors" begin
    backend = TTNKit.ITensorsBackend()
    n_layers = 2
    net = TTNKit.BinaryChainNetwork(n_layers; backend = backend)
    ttn = TTNKit.RandomTreeTensorNetwork(net)
    tensors = vcat(ttn.data...)

    inds = map(p -> TTNKit.internal_index_of_legs(net, p), TTNKit.NodeIterator(net))
    unique_inds, cntr_tn = TTNKit.contract_tensors(tensors, inds)


    res_normal = (tensors[1]*tensors[2])*tensors[3]

    @test res_normal ≈ cntr_tn

end