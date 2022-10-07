using TTNKit, TensorKit
using Test

@testset "Basic Network" begin
    n_layers  = 2
    local_dim = 2
    net = TTNKit.CreateBinaryChainNetwork(n_layers, local_dim)
    @test TTNKit.dimensionality(net) == 1
    
    total_tens = 0
    for jj in 1:n_layers
        total_tens += 2^(jj-1)
        @test TTNKit.lattice(net,jj) == TTNKit.Chain(2^(n_layers-jj), TTNKit.Node)
        @test TTNKit.number_of_tensors(net, jj) == 2^(n_layers-jj)
    end
    @test TTNKit.number_of_tensors(net) == total_tens
    
    @test TTNKit.physical_lattice(net) == TTNKit.Chain(2^n_layers, local_dim)
    @test TTNKit.number_of_layers(net) == n_layers
    
    @test Matrix(TTNKit.adjacency_matrix(net, 1)) == [1 1]
    
    @test TTNKit.node(net, (1,1)) == Node(1, "1")
    
    @test TTNKit.hilbertspace(net, (1,1), local_dim) == ℂ^local_dim
    
    @test TTNKit.dimensions(net,1) == (2,)
    
    @test TTNKit.number_of_sites(net) == 2^n_layers
    @test TTNKit.physical_coordinates(net) == map(x -> Tuple(x), 1:2^n_layers)
    
    @test spacetype(net)  == ComplexSpace
    @test sectortype(net) == Trivial
    @test_throws BoundsError TTNKit.check_valid_position(net, (n_layers+1,1))
    @test TTNKit.index_of_child(net, (0,1)) == 1
    @test TTNKit.index_of_child(net, (0,2)) == 2
    @test TTNKit.index_of_child(net, (0,3)) == 1
    @test TTNKit.index_of_child(net, (0,4)) == 2
    @test TTNKit.index_of_child(net, (1,1)) == 1
    @test TTNKit.index_of_child(net, (1,2)) == 2
    
    @test TTNKit.eachlayer(net) == 1:n_layers
    for ll in 0:n_layers
        @test TTNKit.eachindex(net, ll) == 1:2^(n_layers-ll)
    end
    
    for ll in 0:n_layers-1
        for pp in 1:TTNKit.number_of_tensors(net,ll)
            @test TTNKit.parent_node(net, (ll,pp)) ==  (ll+1, div(pp+1,2))
        end
    end
    @test TTNKit.parent_node(net, (n_layers,1)) == nothing
    
    for ll in TTNKit.eachlayer(net)
        for pp in 1:TTNKit.number_of_tensors(net, ll)
            @test TTNKit.child_nodes(net, (ll, pp)) == [(ll-1, 2*pp-1),(ll-1, 2*pp)]
            @test TTNKit.number_of_child_nodes(net,(ll,pp)) == 2
        end
    end
    
    for pp in 1:TTNKit.number_of_sites(net)
        @test isnothing(TTNKit.child_nodes(net,(0,pp)))
    end
    @test_throws BoundsError TTNKit.parent_node(net, (n_layers+1,1))
    @test_throws BoundsError TTNKit.child_nodes(net, (n_layers+1,1))
    
    @test TTNKit.connecting_path(net, (0,1), (0,4)) == [(1,1),(2,1),(1,2),(0,4)]
    @test TTNKit.connecting_path(net, (1,1), (2,1)) == [(2,1)]
    @test TTNKit.connecting_path(net, (2,1), (1,1)) == [(1,1)]
    @test TTNKit.connecting_path(net, (1,1), (1,2)) == [(2,1), (1,2)]
    @test TTNKit.connecting_path(net, (1,2), (1,1)) == [(2,1), (1,1)]
    
    @test_throws BoundsError TTNKit.connecting_path(net, (-1,1), (1,1))
    @test_throws BoundsError TTNKit.connecting_path(net, (1,1), (-1,1))
    
    vec = Vector{Tuple{Int,Int}}(undef, TTNKit.number_of_tensors(net))
    for (jj, pp) in enumerate(net)
        vec[jj] = pp
    end
    @test vec == [(1,1),(1,2), (2,1)]
    vec = Vector{Tuple{Int,Int}}(undef, TTNKit.number_of_tensors(net))
    for (jj, pp) in enumerate(Iterators.reverse(net))
        vec[jj] = pp
    end
    @test vec == [(2,1),(1,2), (1,1)]
end


@testset "Binary Networks" begin
    n_layers = 2
    
    @test_throws TTNKit.DimensionsException BinaryNetwork((2,3), TrivialNode)


    n_x = 2
    n_y = 4
    @test_throws TTNKit.NotSupportedException TTNKit.BinaryNetwork((n_x, n_y), TrivialNode)
end

@testset "Trivial Constructer" begin
    n_layers = 2
    net = TTNKit.BinaryNetwork((2,))
    # warum schlaegt der fehl?
    #@test net == BinaryNetwork((2,),TrivialNode)
    total_tens = 0
    for pp in net
        @test TTNKit.number_of_child_nodes(net, pp) == 2
        total_tens += 1
    end
    @test TTNKit.number_of_tensors(net) == total_tens
end
    
@testset "D = 1" begin
    n_layers = 2
    D = 1

    net = BinaryChainNetwork(n_layers, TTNKit.HardCoreBosonNode)
    @test TTNKit.number_of_layers(net) == n_layers
    @test TTNKit.number_of_sites(net)  == 2^n_layers
    @test TTNKit.dimensionality(net) == 1
    @test TTNKit.nodetype(TTNKit.lattice(net,1)) == 
                    TTNKit.Node{GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}, U1Irrep}
    
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
    net2 = TTNKit.BinaryChainNetwork((n_sites,), TTNKit.HardCoreBosonNode)
    @test TTNKit.number_of_layers(net2) == n_layers
    @test TTNKit.number_of_sites(net2)  == n_sites


end
    
@testset "D = 2" begin
    n_layers = 2

    n_x = 2^(div(n_layers + 1, 2))
    n_y = 2^(div(n_layers, 2))

    net = BinaryRectangularNetwork(n_layers)
    
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