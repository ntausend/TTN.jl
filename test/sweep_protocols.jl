using TensorKit, TTNKit
using ITensors
using KrylovKit
using Test

@testset "SweepProtocol" begin
    @testset "Simple Sweep Protocol" begin
        # generate a simple network
        n_layers = 3
        n_sweeps = 2
        net = TTNKit.BinaryChainNetwork(n_layers)
        sp  = TTNKit.SimpleSweepHandler(net, n_sweeps)

        @test TTNKit.start_position(sp) == (1,1)
        @test TTNKit.next_position(sp, (1,1))  == (1,2)
        @test TTNKit.number_of_sweeps(sp) == n_sweeps
        @test TTNKit.sweeps(sp) == 1:n_sweeps

        expected_pos = [
            (1,1),
            (1,2),
            (1,3),
            (1,4),
            (2,1),
            (2,2),
            (3,1),
            (2,2),
            (2,1),
            (1,4),
            (1,3),
            (1,2),
            (1,1)
        ]
        
        @test sp.current_sweep == 1
        for (s, sw) in enumerate(TTNKit.sweeps(sp))
            for (jj,pos) in enumerate(sp)
                @test pos == expected_pos[jj]
            end
            @test sp.current_sweep == s + 1
        end
    end

    @testset "TDVPSweepProtocol" begin

    observablesExact1D = [1.0, 0.97185, 0.90716, 0.84981, 0.83449, 0.86299, 0.90798, 0.93797, 0.94063, 0.9263, 0.91295, 0.5159]
    observablesExact2D = [1.0, 0.96257, 0.87714, 0.79915, 0.76395, 0.76467, 0.77569, 0.78988, 0.82486, 0.89178, 0.9657]
    (J, g) = (-1, -2)

    dt = 1e-1
    tmax = 1;

    # Krylov Parameters
    eigsolve_tol = 1e-12
    eigsolve_krylovdim = 30
    eigsolve_maxiter = 3
    eigsolve_verbosity = 0
    ishermitian = true
    eager = true

    func = (action, dt, T) -> exponentiate(action, -1im*dt, T, krylovdim = eigsolve_krylovdim,
                                                               tol = eigsolve_tol, 
                                                               maxiter = eigsolve_maxiter,
                                                               ishermitian = ishermitian,
                                                               eager = eager);  

        @testset "TensorKit: 1D" begin

            L = (4,)
            net = TTNKit.BinaryNetwork(L, TTNKit.SpinHalfNode)
            lat = TTNKit.physical_lattice(net)

            states = fill("Up", TTNKit.number_of_sites(net))
            ttn = TTNKit.ProductTreeTensorNetwork(net, states)
            ttn = TTNKit.increase_dim_tree_tensor_network_zeros(ttn, maxdim = 16)

            ising = TTNKit.TransverseFieldIsing(J = J, g = g);
            tpo = TTNKit.Hamiltonian(ising, lat);
            
            ptpo = TTNKit.ProjTensorProductOperator(ttn, tpo);
            tdvp = TTNKit.TDVPSweepHandler(ttn, ptpo, dt, tmax, func);
            
            observablesTest = []
            σ_z = TensorMap([1 0; 0 -1], ℂ^2 ← ℂ^2) 

            for sw in TTNKit.sweeps(tdvp) 
                append!(observablesTest, [real(sum(TTNKit.expect(ttn, σ_z))/TTNKit.number_of_sites(net))])
                for pos in tdvp
                    TTNKit.update!(tdvp, pos)
                end
                tdvp.current_time += tdvp.timestep;
            end

            for (obTest, obExact) in zip(observablesTest, observablesExact1D)
                @test round(obTest, digits=5) ≈ obExact
            end
        end

        @testset "TensorKit: 2D" begin
            
            L = (2,2)
            net = TTNKit.BinaryNetwork(L, TTNKit.SpinHalfNode)
            lat = TTNKit.physical_lattice(net)

            states = fill("Up", TTNKit.number_of_sites(net))
            ttn = TTNKit.ProductTreeTensorNetwork(net, states)
            ttn = TTNKit.increase_dim_tree_tensor_network_zeros(ttn, maxdim = 16)

            ising = TTNKit.TransverseFieldIsing(J = J, g = g);

            tpo = TTNKit.Hamiltonian(ising, lat, mapping = TTNKit.hilbert_curve(lat));
            
            ptpo = TTNKit.ProjTensorProductOperator(ttn, tpo);
            tdvp = TTNKit.TDVPSweepHandler(ttn, ptpo, dt, tmax, func);
            
            observablesTest = []

            for sw in TTNKit.sweeps(tdvp) 
            σ_z = TensorMap([1 0; 0 -1], ℂ^2 ← ℂ^2) 

            for sw in TTNKit.sweeps(tdvp) 
                append!(observablesTest, [real(sum(TTNKit.expect(ttn, σ_z))/TTNKit.number_of_sites(net))])
                for pos in tdvp
                    TTNKit.update!(tdvp, pos)
                end
                tdvp.current_time += tdvp.timestep;
            end

            for (obTest, obExact) in zip(observablesTest, observablesExact2D)
                @test round(obTest, digits=5) ≈ obExact
            end
        end
    end
        @testset "ITensors: 1D" begin

            L = (4,)
            ind = ITensors.siteinds("S=1/2", L[1])
            net = TTNKit.BinaryNetwork(L, ind)
            lat = TTNKit.physical_lattice(net)

            states = fill("Up", TTNKit.number_of_sites(net))
            ttn = TTNKit.ProductTreeTensorNetwork(net, states)
            ttn = TTNKit.increase_dim_tree_tensor_network_zeros(ttn, maxdim = 16)

            ising = TTNKit.TransverseFieldIsing(J = J, g = g);
            tpo = TTNKit.Hamiltonian(ising, lat);
            
            ptpo = TTNKit.ProjTensorProductOperator(ttn, tpo);
            tdvp = TTNKit.TDVPSweepHandler(ttn, ptpo, dt, tmax, func);
            
            observablesTest = []

            for sw in TTNKit.sweeps(tdvp) 
                append!(observablesTest, [real(sum(TTNKit.expect(ttn, "Z"))/TTNKit.number_of_sites(net))])
                for pos in tdvp
                    TTNKit.update!(tdvp, pos)
                end
                tdvp.current_time += tdvp.timestep;
            end

            for (obTest, obExact) in zip(observablesTest, observablesExact1D)
                @test round(obTest, digits=5) ≈ obExact
            end
        end

        @testset "ITensors: 2D" begin
            
            L = (2,2)
            ind = ITensors.siteinds("S=1/2",prod(L))
            net = TTNKit.BinaryNetwork(L, ind)
            lat = TTNKit.physical_lattice(net)

            states = fill("Up", TTNKit.number_of_sites(net))
            ttn = TTNKit.ProductTreeTensorNetwork(net, states)
            ttn = TTNKit.increase_dim_tree_tensor_network_zeros(ttn, maxdim = 16)

            ising = TTNKit.TransverseFieldIsing(J = J, g = g);

            tpo = TTNKit.Hamiltonian(ising, lat, mapping = TTNKit.hilbert_curve(lat));
            
            ptpo = TTNKit.ProjTensorProductOperator(ttn, tpo);
            tdvp = TTNKit.TDVPSweepHandler(ttn, ptpo, dt, tmax, func);
            
            observablesTest = []

            for sw in TTNKit.sweeps(tdvp) 
                append!(observablesTest, [real(sum(TTNKit.expect(ttn, "Z"))/TTNKit.number_of_sites(net))])
                for pos in tdvp
                    TTNKit.update!(tdvp, pos)
                end
                tdvp.current_time += tdvp.timestep;
            end

            for (obTest, obExact) in zip(observablesTest, observablesExact2D)
                @test round(obTest, digits=5) ≈ obExact
            end
        end
    end
end
