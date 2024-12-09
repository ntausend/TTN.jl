using TTN, ITensors, ITensorMPS
using Test

@testset "Basic Network" begin
    n_layers  = 2
    local_dim = 2
    net = TTN.CreateBinaryChainNetwork(n_layers, local_dim)
    @test TTN.dimensionality(net) == 1
    
    total_tens = 0
    for jj in 1:n_layers
        total_tens += 2^(jj-1)
        @test TTN.lattice(net,jj) == TTN.Chain(2^(n_layers-jj), TTN.Node)
        @test TTN.number_of_tensors(net, jj) == 2^(n_layers-jj)
    end
    @test TTN.number_of_tensors(net) == total_tens
    
    physlat = TTN.physical_lattice(net)
    @test physlat == TTN.Chain(siteinds(physlat))

    hilb = TTN.hilbertspace(net, (1,1), local_dim) 
    hilb_ex = Index(ITensors.id(hilb), local_dim, dir(hilb), tags(hilb), plev(hilb))

    @assert hilb == hilb_ex

    @test TTN.number_of_layers(net) == n_layers
    
    @test Matrix(TTN.adjacency_matrix(net, 1)) == [1 1]
    
    @test TTN.node(net, (1,1)) == TTN.Node(1, "1")
    

    @test TTN.dimensions(net,1) == (2,)
    
    @test TTN.number_of_sites(net) == 2^n_layers
    @test TTN.physical_coordinates(net) == map(x -> Tuple(x), 1:2^n_layers)

    @test TTN.spacetype(net)  == Index
    @test TTN.sectortype(net) == Int64
    @test_throws BoundsError TTN.check_valid_position(net, (n_layers+1,1))
    @test TTN.index_of_child(net, (0,1)) == 1
    @test TTN.index_of_child(net, (0,2)) == 2
    @test TTN.index_of_child(net, (0,3)) == 1
    @test TTN.index_of_child(net, (0,4)) == 2
    @test TTN.index_of_child(net, (1,1)) == 1
    @test TTN.index_of_child(net, (1,2)) == 2
    
    @test TTN.eachlayer(net) == 1:n_layers
    for ll in 0:n_layers
        @test TTN.eachindex(net, ll) == 1:2^(n_layers-ll)
    end
    
    for ll in 0:n_layers-1
        for pp in 1:TTN.number_of_tensors(net,ll)
            @test TTN.parent_node(net, (ll,pp)) ==  (ll+1, div(pp+1,2))
        end
    end
    @test isnothing(TTN.parent_node(net, (n_layers,1)))
    
    for ll in TTN.eachlayer(net)
        for pp in 1:TTN.number_of_tensors(net, ll)
            @test TTN.child_nodes(net, (ll, pp)) == [(ll-1, 2*pp-1),(ll-1, 2*pp)]
            @test TTN.number_of_child_nodes(net,(ll,pp)) == 2
        end
    end
    
    for pp in 1:TTN.number_of_sites(net)
        @test isnothing(TTN.child_nodes(net,(0,pp)))
    end
    @test_throws BoundsError TTN.parent_node(net, (n_layers+1,1))
    @test_throws BoundsError TTN.child_nodes(net, (n_layers+1,1))
    
    @test TTN.connecting_path(net, (0,1), (0,4)) == [(1,1),(2,1),(1,2),(0,4)]
    @test TTN.connecting_path(net, (1,1), (2,1)) == [(2,1)]
    @test TTN.connecting_path(net, (2,1), (1,1)) == [(1,1)]
    @test TTN.connecting_path(net, (1,1), (1,2)) == [(2,1), (1,2)]
    @test TTN.connecting_path(net, (1,2), (1,1)) == [(2,1), (1,1)]
    
    @test_throws BoundsError TTN.connecting_path(net, (-1,1), (1,1))
    @test_throws BoundsError TTN.connecting_path(net, (1,1), (-1,1))
    
    vec = Vector{Tuple{Int,Int}}(undef, TTN.number_of_tensors(net))
    for (jj, pp) in enumerate(TTN.NodeIterator(net))
        vec[jj] = pp
    end
    @test vec == [(1,1),(1,2), (2,1)]
    vec = Vector{Tuple{Int,Int}}(undef, TTN.number_of_tensors(net))
    for (jj, pp) in enumerate(Iterators.reverse(TTN.NodeIterator(net)))
        vec[jj] = pp
    end
    @test vec == [(2,1),(1,2), (1,1)]


    @test TTN.split_index(net, (1,2), 1) == (Tuple(1), (2,3))
    @test TTN.split_index(net, (1,2), 2) == (Tuple(2), (1,3))
    @test TTN.split_index(net, (1,2), 3) == (Tuple(3), (1,2))

    @test TTN.internal_index_of_legs(net, (1,1)) == [1,2,5]
    @test TTN.internal_index_of_legs(net, (1,2)) == [3,4,6]
    @test TTN.internal_index_of_legs(net, (2,1)) == [5,6,7]
end
