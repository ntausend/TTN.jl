using TTNKit
using Test

@testset "Projected MPO, no qn conservation" begin
    conserve_qns = false
    n_layers = 3
    net = TTNKit.BinaryChainNetwork(n_layers, "S=1/2"; conserve_szparity = conserve_qns)
    lat = TTNKit.physical_lattice(net)
    maxBondDim = 4

    J = -1.
    g = -2.
    
    ampo = OpSum()
    for j in eachindex(lat)
        j<length(lat) && (ampo += (J, "X", j, "X", j+1))
        ampo += (g, "Z", j)
    end
    
    
    tpo = TTNKit.Hamiltonian(ampo, lat)

    states = fill("Up", TTNKit.number_of_sites(net))
    ttn = TTNKit.ProductTreeTensorNetwork(net, states, orthogonalize = true)

    ptpo = TTNKit.ProjMPO(ttn, tpo)
    energy_expected = g*TTNKit.number_of_sites(net)
    
    for pos in TTNKit.NodeIterator(net)

        action = TTNKit.∂A(ptpo, pos)
        energy = dot(ttn[pos], action(ttn[pos]))
        @test energy ≈ energy_expected
    end
end        

@testset "Projected MPO, qn conservation" begin
    conserve_qns = true
    n_layers = 3
    net = TTNKit.BinaryChainNetwork(n_layers, "S=1/2"; conserve_szparity = conserve_qns)
    lat = TTNKit.physical_lattice(net)
    maxBondDim = 4

    J = -1.
    g = -2.
    
    ampo = OpSum()
    for j in eachindex(lat)
        j<length(lat) && (ampo += (J, "X", j, "X", j+1))
        ampo += (g, "Z", j)
    end
    
    
    tpo = TTNKit.Hamiltonian(ampo, lat)

    states = fill("Up", TTNKit.number_of_sites(net))
    ttn = TTNKit.ProductTreeTensorNetwork(net, states, orthogonalize = true)

    ptpo = TTNKit.ProjMPO(ttn, tpo)
    energy_expected = g*TTNKit.number_of_sites(net)
    
    for pos in TTNKit.NodeIterator(net)

        action = TTNKit.∂A(ptpo, pos)
        energy = dot(ttn[pos], action(ttn[pos]))
        @test energy ≈ energy_expected
    end
end

@testset "Projected TPO, no qn conservation" begin
    n_layers = 3
    net = TTNKit.BinaryChainNetwork(n_layers, "S=1/2"; conserve_szparity = false)
    lat = TTNKit.physical_lattice(net)
    maxBondDim = 4

    J = -1.
    g = -2.
    
    ampo = OpSum()
    for j in eachindex(lat)
        j<length(lat) && (ampo += (J, "X", j, "X", j+1))
        ampo += (g, "Z", j)
    end
    
    tpo = TTNKit.TPO(ampo, lat)

    states = fill("Up", TTNKit.number_of_sites(net))
    ttn = TTNKit.ProductTreeTensorNetwork(net, states, orthogonalize = true)

    ptpo = TTNKit.ProjTPO(ttn, tpo)
    energy_expected = g*TTNKit.number_of_sites(net)
    
    for pos in TTNKit.NodeIterator(net)
        action = TTNKit.∂A(ptpo, pos)
        energy = dot(ttn[pos], action(ttn[pos]))
        @test energy ≈ energy_expected
    end
end

@testset "Projected TPO, qn conservation" begin
    n_layers = 3
    net = TTNKit.BinaryChainNetwork(n_layers, "S=1/2"; conserve_szparity = true)
    lat = TTNKit.physical_lattice(net)
    maxBondDim = 4

    J = -1.
    g = -2.
    
    ampo = OpSum()
    for j in eachindex(lat)
        j<length(lat) && (ampo += (J, "X", j, "X", j+1))
        ampo += (g, "Z", j)
    end
    
    tpo = TTNKit.TPO(ampo, lat)

    states = fill("Up", TTNKit.number_of_sites(net))
    ttn = TTNKit.ProductTreeTensorNetwork(net, states, orthogonalize = true)

    ptpo = TTNKit.ProjTPO(ttn, tpo)
    energy_expected = g*TTNKit.number_of_sites(net)
    
    for pos in TTNKit.NodeIterator(net)
        
        action = TTNKit.∂A(ptpo, pos)

        energy = dot(ttn[pos], action(ttn[pos]))
        @test energy ≈ energy_expected
    end
end