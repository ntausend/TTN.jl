using TensorKit, TTNKit
using Test

function n_op_non_sym(number_of_bosons)
    n_total = number_of_bosons + 1
    sp = ℂ^(n_total)
    mat = diagm(collect(0:number_of_bosons))
    return TensorMap(mat, sp ← sp)
end

function n_op_sym(number_of_bosons)
    # gettting the hilbertspace by using the SoftCoreBosonNode Hilbertspace
    sp = hilbertspace(SCB(1; conserve_qns = true, number_of_bosons = number_of_bosons) )
    mpo = TensorMap(zeros, Float64, sp, sp)
    for se in sectors(sp)
        blocks(mpo)[se] .= [se.charge]
    end
    return mpo
end
n_op(number_of_bosons; conserve_qns = true) = conserve_qns ? n_op_sym(number_of_bosons) : n_op_non_sym(number_of_bosons)

function sig_z_non_sym()
    σ_z = [0 0; 0 1]
    sp  = ℂ^2
    return TensorMap(σ_z, sp ← sp)
end

function sig_z_sym()
    sp_u1 = U1Space(0 => 1, 1 => 1)
    mpo = TensorMap(zeros, Float64, sp_u1, sp_u1)
    blocks(mpo)[Irrep[U₁](0)] .= [0]
    blocks(mpo)[Irrep[U₁](1)] .= [1]   
    
    return mpo
end
sig_z(;conserve_qns = true) = conserve_qns ? sig_z_sym() : sig_z_non_sym()

function test_expectation(dims, conserve_qns, op, ndtype, states, expected)
    net = TTNKit.BinaryNetwork(dims, ndtype; conserve_qns = conserve_qns)
    ttn = TTNKit.ProductTreeTensorNetwork(net, states)
    obs_measured = TTNKit.expect(ttn, op)
    @test all(obs_measured .≈ expected)
end


@testset "Non-symmetric, HCB, 1D, TensorKit" begin
    n_sites = 16
    conserve_qns = false
    dims = Tuple(n_sites)
    expected = rand(0:1, n_sites)
    states = map(expected) do s
        return s == 0 ? "Emp" : "Occ"
    end
    op = sig_z(;conserve_qns = conserve_qns)
    test_expectation(dims, conserve_qns, op, TTNKit.HardCoreBosonNode, states, expected)
end

@testset "Value, symmetric, HCB, 1D, TensorKit" begin
    n_sites = 16
    conserve_qns = true
    dims = Tuple(n_sites)
    expected = rand(0:1, n_sites)
    states = map(expected) do s
        return s == 0 ? "Emp" : "Occ"
    end
    op = sig_z(;conserve_qns = conserve_qns)
    test_expectation(dims, conserve_qns, op, TTNKit.HardCoreBosonNode, states, expected)
end

@testset "Non-symmetric, HCB, 2D, square lattice, TensorKit" begin
    nx = 4
    ny = 4
    n_sites = nx*ny
    
    conserve_qns = false
    dims = (nx, ny)
    expected = rand(0:1, n_sites)
    states = map(expected) do s
        return s == 0 ? "Emp" : "Occ"
    end
    expected = reshape(expected, dims)

    op = sig_z(;conserve_qns = conserve_qns)
    test_expectation(dims, conserve_qns, op, TTNKit.HardCoreBosonNode, states, expected)
end

@testset "Non-symmetric, HCB, 2D, non-square lattice (x larger), TensorKit" begin
    nx = 4
    ny = 2
    n_sites = nx*ny
    
    conserve_qns = false
    dims = (nx, ny)
    expected = rand(0:1, n_sites)
    states = map(expected) do s
        return s == 0 ? "Emp" : "Occ"
    end
    expected = reshape(expected, dims)

    op = sig_z(;conserve_qns = conserve_qns)
    test_expectation(dims, conserve_qns, op, TTNKit.HardCoreBosonNode, states, expected)
end

#= This is not supported, we need x≥y because of the pairing pattern...
@testset "Non-symmetric, HCB, 2D, non-square lattice (y larger), TensorKit" begin
    nx = 2
    ny = 4
    n_sites = nx*ny
    
    conserve_qns = false
    dims = (nx, ny)
    expected = rand(0:1, n_sites)
    states = map(expected) do s
        return s == 0 ? "Emp" : "Occ"
    end
    expected = reshape(expected, dims)

    op = sig_z(;conserve_qns = conserve_qns)
    test_expectation(dims, conserve_qns, op, TTNKit.HardCoreBosonNode, states, expected)
end
=#

@testset "Symmetric, HCB, 2D, non-square lattice (x larger), TensorKit" begin
    nx = 4
    ny = 2
    n_sites = nx*ny
    
    conserve_qns = true
    dims = (nx, ny)
    expected = rand(0:1, n_sites)
    states = map(expected) do s
        return s == 0 ? "Emp" : "Occ"
    end
    expected = reshape(expected, dims)

    op = sig_z(;conserve_qns = conserve_qns)
    test_expectation(dims, conserve_qns, op, TTNKit.HardCoreBosonNode, states, expected)
end

#= This is not supported, we need x≥y because of the pairing pattern...
@testset "Symmetric, HCB, 2D, non-square lattice (y larger), TensorKit" begin
    nx = 2
    ny = 4
    n_sites = nx*ny
    
    conserve_qns = true
    dims = (nx, ny)
    expected = rand(0:1, n_sites)
    states = map(expected) do s
        return s == 0 ? "Emp" : "Occ"
    end
    expected = reshape(expected, dims)

    op = sig_z(;conserve_qns = conserve_qns)
    test_expectation(dims, conserve_qns, op, TTNKit.HardCoreBosonNode, states, expected)
end
=#