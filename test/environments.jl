using TTN
using Test

@testset "Projected MPO, no qn conservation" begin
    conserve_qns = false
    n_layers = 3
    net = TTN.BinaryChainNetwork(n_layers, "S=1/2"; conserve_szparity = conserve_qns)
    lat = TTN.physical_lattice(net)
    maxBondDim = 4

    J = -1.
    g = -2.
    
    ampo = OpSum()
    for j in eachindex(lat)
        j<length(lat) && (ampo += (J, "X", j, "X", j+1))
        ampo += (g, "Z", j)
    end
    
    
    tpo = TTN.Hamiltonian(ampo, lat)

    states = fill("Up", TTN.number_of_sites(net))
    ttn = TTN.ProductTreeTensorNetwork(net, states, orthogonalize = true)

    ptpo = TTN.ProjMPO(ttn, tpo)
    energy_expected = g*TTN.number_of_sites(net)
    
    for pos in TTN.NodeIterator(net)

        action = TTN.∂A(ptpo, pos)
        energy = dot(ttn[pos], action(ttn[pos]))
        @test energy ≈ energy_expected
    end
end        

@testset "Projected MPO, qn conservation" begin
    conserve_qns = true
    n_layers = 3
    net = TTN.BinaryChainNetwork(n_layers, "S=1/2"; conserve_szparity = conserve_qns)
    lat = TTN.physical_lattice(net)
    maxBondDim = 4

    J = -1.
    g = -2.
    
    ampo = OpSum()
    for j in eachindex(lat)
        j<length(lat) && (ampo += (J, "X", j, "X", j+1))
        ampo += (g, "Z", j)
    end
    
    
    tpo = TTN.Hamiltonian(ampo, lat)

    states = fill("Up", TTN.number_of_sites(net))
    ttn = TTN.ProductTreeTensorNetwork(net, states, orthogonalize = true)

    ptpo = TTN.ProjMPO(ttn, tpo)
    energy_expected = g*TTN.number_of_sites(net)
    
    for pos in TTN.NodeIterator(net)

        action = TTN.∂A(ptpo, pos)
        energy = dot(ttn[pos], action(ttn[pos]))
        @test energy ≈ energy_expected
    end
end

@testset "Projected TPO, no qn conservation" begin
    n_layers = 3
    net = TTN.BinaryChainNetwork(n_layers, "S=1/2"; conserve_szparity = false)
    lat = TTN.physical_lattice(net)
    maxBondDim = 4

    J = -1.
    g = -2.
    
    ampo = OpSum()
    for j in eachindex(lat)
        j<length(lat) && (ampo += (J, "X", j, "X", j+1))
        ampo += (g, "Z", j)
    end
    
    tpo = TTN.TPO(ampo, lat)

    states = fill("Up", TTN.number_of_sites(net))
    ttn = TTN.ProductTreeTensorNetwork(net, states, orthogonalize = true)

    ptpo = TTN.ProjTPO(ttn, tpo)
    energy_expected = g*TTN.number_of_sites(net)
    
    for pos in TTN.NodeIterator(net)
        action = TTN.∂A(ptpo, pos)
        energy = dot(ttn[pos], action(ttn[pos]))
        @test energy ≈ energy_expected
    end
end

@testset "Projected TPO, qn conservation" begin
    n_layers = 3
    net = TTN.BinaryChainNetwork(n_layers, "S=1/2"; conserve_szparity = true)
    lat = TTN.physical_lattice(net)
    maxBondDim = 4

    J = -1.
    g = -2.
    
    ampo = OpSum()
    for j in eachindex(lat)
        j<length(lat) && (ampo += (J, "X", j, "X", j+1))
        ampo += (g, "Z", j)
    end
    
    tpo = TTN.TPO(ampo, lat)

    states = fill("Up", TTN.number_of_sites(net))
    ttn = TTN.ProductTreeTensorNetwork(net, states, orthogonalize = true)

    ptpo = TTN.ProjTPO(ttn, tpo)
    energy_expected = g*TTN.number_of_sites(net)
    
    for pos in TTN.NodeIterator(net)
        
        action = TTN.∂A(ptpo, pos)

        energy = dot(ttn[pos], action(ttn[pos]))
        @test energy ≈ energy_expected
    end
end
