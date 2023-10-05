using TTNKit, ITensors
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
    
    physlat = TTNKit.physical_lattice(net)
    @test physlat == TTNKit.Chain(siteinds(physlat))

    hilb = TTNKit.hilbertspace(net, (1,1), local_dim) 
    hilb_ex = Index(ITensors.id(hilb), local_dim, dir(hilb), tags(hilb), plev(hilb))

    @assert hilb == hilb_ex

    @test TTNKit.number_of_layers(net) == n_layers
    
    @test Matrix(TTNKit.adjacency_matrix(net, 1)) == [1 1]
    
    @test TTNKit.node(net, (1,1)) == TTNKit.Node(1, "1")
    

    @test TTNKit.dimensions(net,1) == (2,)
    
    @test TTNKit.number_of_sites(net) == 2^n_layers
    @test TTNKit.physical_coordinates(net) == map(x -> Tuple(x), 1:2^n_layers)

    @test TTNKit.spacetype(net)  == Index
    @test TTNKit.sectortype(net) == Int64
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
    @test isnothing(TTNKit.parent_node(net, (n_layers,1)))
    
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
    for (jj, pp) in enumerate(TTNKit.NodeIterator(net))
        vec[jj] = pp
    end
    @test vec == [(1,1),(1,2), (2,1)]
    vec = Vector{Tuple{Int,Int}}(undef, TTNKit.number_of_tensors(net))
    for (jj, pp) in enumerate(Iterators.reverse(TTNKit.NodeIterator(net)))
        vec[jj] = pp
    end
    @test vec == [(2,1),(1,2), (1,1)]


    @test TTNKit.split_index(net, (1,2), 1) == (Tuple(1), (2,3))
    @test TTNKit.split_index(net, (1,2), 2) == (Tuple(2), (1,3))
    @test TTNKit.split_index(net, (1,2), 3) == (Tuple(3), (1,2))

    @test TTNKit.internal_index_of_legs(net, (1,1)) == [1,2,5]
    @test TTNKit.internal_index_of_legs(net, (1,2)) == [3,4,6]
    @test TTNKit.internal_index_of_legs(net, (2,1)) == [5,6,7]
end