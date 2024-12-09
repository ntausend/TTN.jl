using Test

function test_expectation(dims, conserve_qns, op, ndtype, states, expected)
    net = TTN.BinaryNetwork(dims, ndtype; conserve_qns = conserve_qns)
    ttn = TTN.ProductTreeTensorNetwork(net, states)
    obs_measured = real.(TTN.expect(ttn, op))
    @test all(obs_measured .≈ expected)
end


@testset "Non-symmetric, SpinHalf, 1D, ITensors" begin
    n_sites = 16
    conserve_qns = false
    dims = Tuple(n_sites)
    expected = rand(0:1, n_sites).*2 .- 1
    states = map(expected) do s
        return s == 1 ? "Up" : "Dn"
    end
    test_expectation(dims, conserve_qns, "Z", "SpinHalf", states, expected)
end

@testset "Value, symmetric, SpinHalf, 1D, ITensors" begin
    n_sites = 16
    conserve_qns = true
    dims = Tuple(n_sites)
    expected = rand(0:1, n_sites).*2 .- 1
    states = map(expected) do s
        return s == 1 ? "Up" : "Dn"
    end
    test_expectation(dims, conserve_qns, "Z", "SpinHalf", states, expected)
end

@testset "Non-symmetric, SpinHaf, 2D, square lattice, ITensors" begin
    nx = 4
    ny = 4
    n_sites = nx*ny
    
    conserve_qns = false
    dims = (nx, ny)
    expected = rand(0:1, n_sites).*2 .- 1
    states = map(expected) do s
        return s == 1 ? "Up" : "Dn"
    end
    expected = reshape(expected, dims)

    test_expectation(dims, conserve_qns, "Z", "SpinHalf", states, expected)
end

@testset "Non-symmetric, SpinHalf, 2D, non-square lattice (x larger), ITensors" begin
    nx = 4
    ny = 2
    n_sites = nx*ny
    
    conserve_qns = false
    dims = (nx, ny)
    expected = rand(0:1, n_sites).*2 .- 1
    states = map(expected) do s
        return s == 1 ? "Up" : "Dn"
    end
    expected = reshape(expected, dims)

    test_expectation(dims, conserve_qns, "Z", "SpinHalf", states, expected)
end

#= This is not supported, we need x≥y because of the pairing pattern...
@testset "Non-symmetric, SpinHalf, 2D, non-square lattice (y larger), ITensors" begin
    nx = 2
    ny = 4
    n_sites = nx*ny
    
    conserve_qns = false
    dims = (nx, ny)
    expected = rand(0:1, n_sites).*2 .- 1
    states = map(expected) do s
        return s == 1 ? "Up" : "Dn"
    end
    expected = reshape(expected, dims)

    test_expectation(dims, conserve_qns, "Z", "SpinHalf", states, expected)
end
=#

@testset "Symmetric, SpinHalf, 2D, non-square lattice (x larger), ITensors" begin
    nx = 4
    ny = 2
    n_sites = nx*ny
    
    conserve_qns = true
    dims = (nx, ny)
    expected = rand(0:1, n_sites).*2 .- 1
    states = map(expected) do s
        return s == 1 ? "Up" : "Dn"
    end
    expected = reshape(expected, dims)

    test_expectation(dims, conserve_qns, "Z", "SpinHalf", states, expected)
end

#= This is not supported, we need x≥y because of the pairing pattern...
@testset "Symmetric, HCB, 2D, non-square lattice (y larger), ITensors" begin
    nx = 2
    ny = 4
    n_sites = nx*ny
    
    conserve_qns = true
    dims = (nx, ny)
    expected = rand(0:1, n_sites).*2 .- 1
    states = map(expected) do s
        return s == 1 ? "Up" : "Dn"
    end
    expected = reshape(expected, dims)

    test_expectation(dims, conserve_qns, "Z", "SpinHalf", states, expected)
end
=#
