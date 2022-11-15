using TTNKit, TensorKit
using Test

@testset "General Tree Tensor Network" begin
    n_layers = 3
    ndtyp = TTNKit.TrivialNode
    net = BinaryChainNetwork(n_layers, ndtyp)
    
    ttn = RandomTreeTensorNetwork(net, orthogonalize = false)
    
    @test TTNKit.number_of_layers(ttn) == n_layers
    @test length(TTNKit.layer(ttn,1))  == 2^(n_layers-1)
    @test TTNKit.network(ttn) == net

    @test TTNKit.ortho_center(ttn) == (-1,-1)
    for p in TTNKit.NodeIterator(net)
        @test TTNKit.ortho_direction(ttn, p) == -1
    end
    
    ttn = RandomTreeTensorNetwork(net, orthogonalize = true)

    @test TTNKit.ortho_center(ttn) == (n_layers,1)
    @test TTNKit.ortho_direction(ttn, (n_layers,1)) == -1
    for p in TTNKit.NodeIterator(net)
        if (p == (n_layers, 1))
            @test TTNKit.ortho_direction(ttn, p) == -1
        else
            @test TTNKit.ortho_direction(ttn, p) == 3
        end
    end

    @test ttn[1,1] == ttn[(1,1)]
    
    n_ten = TensorMap(randn, ℂ^2 ⊗ ℂ^2 ← ℂ^1)
    ttn[1,1] = n_ten
    @test ttn[1,1] == n_ten
    ttn = RandomTreeTensorNetwork(net, orthogonalize = true)

    TTNKit.move_down!(ttn,1)
    @test TTNKit.ortho_center(ttn) == (n_layers-1,1)
    @test TTNKit.ortho_direction(ttn, (n_layers, 1)) == 1
    @test TTNKit.ortho_direction(ttn, (n_layers-1, 1)) == -1


    TTNKit.move_up!(ttn)
    @test TTNKit.ortho_center(ttn) == (n_layers,1)
    @test TTNKit.ortho_direction(ttn, (n_layers, 1))   == -1
    @test TTNKit.ortho_direction(ttn, (n_layers-1, 1)) == 3

    oc = (1,1)
    TTNKit.move_ortho!(ttn, oc)
    @test TTNKit.ortho_center(ttn) == oc
    for p in TTNKit.NodeIterator(net)
        if(p == oc)
            @test TTNKit.ortho_direction(ttn, p) == -1
        else
            c_path = TTNKit.connecting_path(net, p, oc)
            nd_next = c_path[1]
            if nd_next[1] == p[1] - 1
                @test TTNKit.ortho_direction(ttn, p) == nd_next[2]
            else
                @test TTNKit.ortho_direction(ttn, p) == 3
            end
        end
    end
    
    net = BinaryRectangularNetwork(n_layers, TTNKit.HardCoreBosonNode)
    ttn = RandomTreeTensorNetwork(net, orthogonalize = true)

    TTNKit.move_down!(ttn,1)
    @test TTNKit.ortho_center(ttn) == (n_layers-1,1)
    TTNKit.move_up!(ttn)
    @test TTNKit.ortho_center(ttn) == (n_layers,1)
    TTNKit.move_ortho!(ttn, (1,1))
    @test TTNKit.ortho_center(ttn) == (1,1)
    
    is_normal, res = TTNKit.check_normality(ttn)
    @test is_normal
    @test res ≈ 1


    elT = Float64
    ttn = RandomTreeTensorNetwork(net; elT = elT)
    @test eltype(ttn) == elT

    elT = ComplexF64
    ttn = RandomTreeTensorNetwork(net; elT = elT)
    @test eltype(ttn) == elT
end

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

@testset "Generate Random Number conserved State with target charge" begin
    # number of layers
    n_layer = 4
    # linear lattice
    conserve_qns = true
    net = TTNKit.BinaryNetwork((4,4), TTNKit.HardCoreBosonNode; conserve_qns = conserve_qns)

    chrg = 5
    target_charge = conserve_qns ? U1Irrep(chrg) : Trivial()
    maxdim = 16
    
    ttn = TTNKit.RandomTreeTensorNetwork(net, target_charge; maxdim = maxdim)


    # number operator
    n_z = sig_z(; conserve_qns = conserve_qns)

    # expectation value
    n_z_exp = TTNKit.expect(ttn, n_z)
     
    @test sum(n_z_exp) ≈ chrg
end

@testset "Generate Random Number conserved State with random charge" begin
    # number of layers
    n_layer = 4
    # linear lattice
    conserve_qns = true
    net = TTNKit.BinaryNetwork((4,4), TTNKit.HardCoreBosonNode; conserve_qns = conserve_qns)

    maxdim = 16
    
    ttn = TTNKit.RandomTreeTensorNetwork(net; maxdim = maxdim)


    # number operator
    n_z = sig_z(; conserve_qns = conserve_qns)

    # expectation value
    n_z_exp = TTNKit.expect(ttn, n_z)
     
    @test round(Int64, sum(n_z_exp)) ≈ sum(n_z_exp)
end
