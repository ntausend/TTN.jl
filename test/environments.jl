using TTNKit, TensorKit
using Test

@testset "Projected TPO, no qn conservation" begin
    n_layers = 3
    ndtyp = TTNKit.SpinHalfNode
    net = TTNKit.BinaryChainNetwork(n_layers, ndtyp)
    lat = TTNKit.physical_lattice(net)
    maxBondDim = 4

    J = -1.
    g = -2.
    tpo = TTNKit.Hamiltonian(TTNKit.TransverseFieldIsing(J = J, g = g), lat)

    states = fill("Up", TTNKit.number_of_sites(net))
    ttn = TTNKit.ProductTreeTensorNetwork(net, states, orthogonalize = true)

    ptpo = TTNKit.ProjTensorProductOperator(ttn, tpo)
    energy_expected = g*TTNKit.number_of_sites(net)

    
    
    for pos in TTNKit.NodeIterator(net)
        try
            parent = TTNKit.parent_node(net, pos)
            tenv = TTNKit.top_environment(ptpo, pos)
            benv = TTNKit.bottom_environment(ptpo, parent)[1]


            #env = TTNKit.environment(ptpo, p)


            #t = TensorKit.permute(ttn[pos], (1,2,3), ())
            @tensor energy = benv[1,2,3] * tenv[1,2,3]
            @test energy ≈ energy_expected
        catch
            @test_broken false
        end
    end
        


    #=
    statesUp = fill("Up", TTNKit.number_of_sites(net))
    ttnUp = ProductTreeTensorNetwork(net, statesUp, orthogonalize = false)
    ttnUp = increase_dim_tree_tensor_network_randn(ttnUp, maxdim = maxBondDim, factor = 10e-12)

    ptpoUp = ProjTensorProductOperator(ttnUp, tpo)
    energy_test_Up = J*(TTNKit.number_of_sites(net)-1)

    @test TTNKit.number_of_layers(ttnUp) == n_layers
    for ll in TTNKit.eachlayer(net)
        for pp in TTNKit.eachindex(net,ll)
            env = environment(ptpoUp, (ll,pp))
            t = copy(TensorKit.permute(ttnUp[(ll,pp)], (1,2,3), ()))
            energy = (adjoint(t)*env*t)[1]
            @test energy ≈ energy_test_Up
        end
    end
    =#
end        