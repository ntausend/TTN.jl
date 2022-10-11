using TensorKit, TTNKit
using Test

function n_non_sym()
    σ_z = [0 0; 0 1]
    sp  = ℂ^2
    return TensorMap(σ_z, sp ← sp)
end

function n_sym()
    sp_u1 = U1Space(0 => 1, 1 => 1)
    mpo = TensorMap(zeros, Float64, sp_u1, sp_u1)
    blocks(mpo)[Irrep[U₁](0)] .= [0]
    blocks(mpo)[Irrep[U₁](1)] .= [1]   
    
    return mpo
end
n_op(;conserve_qns = true) = conserve_qns ? n_sym() : n_non_sym()

@testset "Product TTN Correlation functions" begin
    n_sites = 16
    conserve_qns = false
    dims = Tuple(n_sites)
    net = TTNKit.BinaryNetwork(dims, TTNKit.HardCoreBosonNode; conserve_qns = conserve_qns)

    states = repeat(["Occ","Emp"], n_sites÷2)

    ttn = TTNKit.ProductTreeTensorNetwork(net, states)

    op = n_op(;conserve_qns = conserve_qns)

    expected_corr = map(states) do s
        s == "Occ" ? 1 : 0
    end

    corr_measured = map(1:n_sites) do jj
        TTNKit.correlation(ttn, op, op, 1,jj)
    end
    @test all(corr_measured .== expected_corr)
end