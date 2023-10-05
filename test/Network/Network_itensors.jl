
using TTNKit, ITensors
using Test


@testset "Binary Networks, ITensors" begin
    n_layers = 2
    
    @test_throws TTNKit.DimensionsException TTNKit.BinaryNetwork((2,3), TTNKit.TrivialNode)

    n_x = 2
    n_y = 4
    @test_throws TTNKit.NotSupportedException TTNKit.BinaryNetwork((n_x, n_y), TTNKit.TrivialNode)

end

@testset "Trivial Constructer, ITensors" begin
    n_layers = 2
    net = TTNKit.BinaryNetwork((2,))

    # warum schlaegt der fehl?
    #@test net == BinaryNetwork((2,),TrivialNode)
    total_tens = 0
    for pp in TTNKit.NodeIterator(net)
        @test TTNKit.number_of_child_nodes(net, pp) == 2
        total_tens += 1
    end
    @test TTNKit.number_of_tensors(net) == total_tens

end
    
@testset "D = 1, ITensors" begin
    n_layers = 2
    D = 1

    net = TTNKit.BinaryChainNetwork(n_layers, "SpinHalf", conserve_qns = true)

    @test TTNKit.internal_index_of_legs(net, (1,1)) == [1,2,5]
    @test TTNKit.internal_index_of_legs(net, (1,2)) == [3,4,6]
    @test TTNKit.internal_index_of_legs(net, (2,1)) == [5,6,7]

    @test TTNKit.number_of_layers(net) == n_layers
    @test TTNKit.number_of_sites(net)  == 2^n_layers
    @test TTNKit.dimensionality(net) == 1
    @test TTNKit.nodetype(TTNKit.lattice(net,1)) == TTNKit.Node{Index, Vector{Pair{QN, Int64}}} 
    
    for ll in 0:n_layers-1
        for pp in 1:TTNKit.number_of_tensors(net,ll)
            @test TTNKit.parent_node(net, (ll,pp)) ==  (ll+1, div(pp+1,2))
        end
    
    end
    @test isnothing(TTNKit.parent_node(net, (n_layers,1)))
    
    for ll in TTNKit.eachlayer(net)
        for pp in 1:TTNKit.number_of_tensors(net, ll)
            @test TTNKit.child_nodes(net, (ll, pp)) == [(ll-1, 2*pp-1),(ll-1, 2*pp)]
            @test TTNKit.number_of_child_nodes(net,(ll,pp)) == 2
        end
    end
    @test TTNKit.index_of_child(net, (0,1)) == 1
    @test TTNKit.index_of_child(net, (0,2)) == 2
    @test TTNKit.index_of_child(net, (0,3)) == 1
    @test TTNKit.index_of_child(net, (0,4)) == 2
    @test TTNKit.index_of_child(net, (1,1)) == 1
    @test TTNKit.index_of_child(net, (1,2)) == 2
    
    @test Matrix(TTNKit.adjacency_matrix(net, 1)) == [1 1]

    n_sites = 4
    net2 = TTNKit.BinaryChainNetwork(TTNKit.siteinds(net))
    @test TTNKit.number_of_layers(net2) == n_layers
    @test TTNKit.number_of_sites(net2)  == n_sites
    net3 = TTNKit.BinaryChainNetwork((n_sites,), TTNKit.ITensorNode, "SpinHalf")
    @test TTNKit.number_of_layers(net3) == n_layers
    @test TTNKit.number_of_sites(net3)  == n_sites

end
    
@testset "D = 2" begin
    n_layers = 2

    n_x = 2^(div(n_layers + 1, 2))
    n_y = 2^(div(n_layers, 2))

    net = TTNKit.BinaryRectangularNetwork(n_layers)
    
    @test size(TTNKit.physical_lattice(net)) == (n_x, n_y)
    
    @test TTNKit.parent_node(net, (0,1)) == (1,1)
    @test TTNKit.parent_node(net, (0,2)) == (1,1)
    @test TTNKit.parent_node(net, (0,3)) == (1,2)
    @test TTNKit.parent_node(net, (1,1)) == (2,1)
    
    @test TTNKit.child_nodes(net, (1,1)) == [(0,1),(0,2)]
    
    @test TTNKit.index_of_child(net, (0,1)) == 1
    @test TTNKit.index_of_child(net, (0,2)) == 2
    
    @test Matrix(TTNKit.adjacency_matrix(net, 1)) == [1 1]

end