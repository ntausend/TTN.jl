function StarOpHam(J, lat; periodic=false)
   ampo = ITensors.OpSum()
   # for p in TTN.coordinates(lat)
   #    ampo += 1.0, "X", p
   # end
   NN = nearest_neighbours(lat, collect(eachindex(lat)); periodic = periodic)
   adj = Dict{Int, Set{Int}}()

   for (i, j) in NN
      push!(get!(adj, i, Set{Int}()), j)
      push!(get!(adj, j, Set{Int}()), i)   # comment for “forward-only”
   end

   sites = sort(collect(keys(adj)))         # ← collect before sort!

   neigh_tuples = [
      (site, sort(collect(adj[site]))...)  # neighbours as an ordered tuple
      for site in sites
   ]

   for bond in neigh_tuples
      if length(bond) == 3
         # println("Adding bond: ", bond[1], " ", bond[2], " ", bond[3])
         b1 = TTN.coordinate(lat, bond[1])
         b2 = TTN.coordinate(lat, bond[2])
         b3 = TTN.coordinate(lat, bond[3])
         ampo += J, "Z", b1, "Z", b2, "Z", b3
      elseif length(bond) == 4
         # println("Adding bond: ", bond[1], " ", bond[2], " ", bond[3], " ", bond[4])
         b1 = TTN.coordinate(lat, bond[1])
         b2 = TTN.coordinate(lat, bond[2])
         b3 = TTN.coordinate(lat, bond[3])
         b4 = TTN.coordinate(lat, bond[4])
         ampo += J, "Z", b1, "Z", b2, "Z", b3, "Z", b4
      elseif length(bond) == 5
         # println("Adding bond: ", bond[1], " ", bond[2], " ", bond[3], " ", bond[4], " ", bond[5])
         b1 = TTN.coordinate(lat, bond[1])
         b2 = TTN.coordinate(lat, bond[2])
         b3 = TTN.coordinate(lat, bond[3])
         b4 = TTN.coordinate(lat, bond[4])
         b5 = TTN.coordinate(lat, bond[5])
         ampo += J, "Z", b1, "Z", b2, "Z", b3, "Z", b4, "Z", b5
      end
   end
   return ampo
end

using Test
using IterTools
@testset "Test up to 3-5-site operators" begin
    dims = (2^2,2^2) # dimensions of the rectangle
    J = 2 # setting the model parameter
    maxdims = 6 # maximal bond dimension of the random tensor network
    # Creates a network of nodes with a local Hilbertspace of S=1/2 spins
    net = BinaryRectangularNetwork(dims, "SpinHalf")

    lat = physical_lattice(net)
    ampo_star = StarOpHam(J, lat; periodic = false);

    ttn0 = RandomTreeTensorNetwork(net);
    # states = fill("Up", TTN.number_of_sites(net))
    # ttn0 = TTN.ProductTreeTensorNetwork(net, states, orthogonalize = true)  

    #Revised TTN.jl calculation
    tpo = build_tpo_from_opsum(ampo_star, lat); # TPO_group
    link_ops = upflow_to_root(net, ttn0, tpo, (4,1));
    test_contr = complete_contraction(net, ttn0, link_ops, (4,1))
    val_ref = Array(test_contr)[1]

    for pos in TTN.NodeIterator(net)
        link_ops = upflow_to_root(net, ttn0, tpo, pos);
        test_contr = complete_contraction(net, ttn0, link_ops, pos)
        val = Array(test_contr)[1]
        println("Position: $pos, value: $val")
        @test val ≈ val_ref
    end
end