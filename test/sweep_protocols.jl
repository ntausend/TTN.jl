using TTN
using ITensors, ITensorMPS
using Test

@testset "SweepProtocol" begin
    # @testset "Simple Sweep Protocol" begin
    #     # generate a simple network
    #     n_layers = 3
    #     n_sweeps = 2
    #     net = TTN.BinaryChainNetwork(n_layers)
    #     sp  = TTN.SimpleSweepHandler(net, n_sweeps)
    #
    #     @test TTN.start_position(sp) == (1,1)
    #     @test TTN.next_position(sp, (1,1))  == (1,2)
    #     @test TTN.number_of_sweeps(sp) == n_sweeps
    #     @test TTN.sweeps(sp) == 1:n_sweeps
    #
    #     expected_pos = [
    #         (1,1),
    #         (1,2),
    #         (1,3),
    #         (1,4),
    #         (2,1),
    #         (2,2),
    #         (3,1),
    #         (2,2),
    #         (2,1),
    #         (1,4),
    #         (1,3),
    #         (1,2),
    #         (1,1)
    #     ]
    #     
    #     @test sp.current_sweep == 1
    #     for (s, sw) in enumerate(TTN.sweeps(sp))
    #         for (jj,pos) in enumerate(sp)
    #             @test pos == expected_pos[jj]
    #         end
    #         @test sp.current_sweep == s + 1
    #     end
    # end

    @testset "TDVPSweepProtocol" begin

        mutable struct testObserver <: AbstractObserver
          energy::Vector{Float64}
          pol::Vector{Float64}

          testObserver() = new(Float64[], Float64[])
        end

        function ITensorMPS.measure!(ob::testObserver; kwargs...)
          tdvp = kwargs[:sweep_handler]
          #pos = kwargs[:pos]

          #((tdvp.dirloop !== :backward) || (pos !== tdvp.path[2])) && return
          (tdvp.dirloop !== :backward) && return

          topPos = (TTN.number_of_layers(TTN.network(tdvp.ttn)), 1)
          n_sites = TTN.number_of_sites(TTN.network(tdvp.ttn))

          action = TTN.∂A(tdvp.pTPO, topPos)
          T = tdvp.ttn[topPos]
          actionT = action(T)

          push!(ob.energy, real(ITensors.scalar(dag(T)*actionT)))
          push!(ob.pol, real(sum(TTN.expect(tdvp.ttn, "Z"))/n_sites))

        end

        function TransverseFieldIsing2d(J,g,lat)
           ampo = OpSum()
           for p in TTN.coordinates(lat)
              ampo += g, "Z", p
           end
           for bond in TTN.nearest_neighbours(lat, collect(eachindex(lat)))
              b1 = TTN.coordinate(lat, first(bond))
              b2 = TTN.coordinate(lat, last(bond))
              ampo += J, "X", b1, "X", b2
           end
           return ampo
        end

        observablesExact1D = [0.97185, 0.90716, 0.84981, 0.83449, 0.86299, 0.90798, 0.93797, 0.94063, 0.9263, 0.91295, 0.9094]
        observablesExact2D = [0.96257, 0.87714, 0.79915, 0.76395, 0.76467, 0.77569, 0.78988, 0.82486, 0.89178, 0.9657]
        (J, g) = (-1, -2)
        (tmax,dt) = (1, 1e-1)


        @testset "1D" begin

            L = (4,)
            ind = ITensors.siteinds("S=1/2", L[1])
            net = TTN.BinaryNetwork(L, ind)
            lat = TTN.physical_lattice(net)

            states = fill("Up", TTN.number_of_sites(net))
            ttn = TTN.ProductTreeTensorNetwork(net, states)
            ttn = TTN.increase_dim_tree_tensor_network_zeros(ttn, maxdim = 16)

            ising = TransverseFieldIsing2d(J, g, lat);
            tpo = TTN.TPO(ising, lat)
            
            obs = testObserver()
            TTN.tdvp(ttn, tpo, finaltime = tmax, timestep = dt, observer = obs)
            
            for (polExact, polTest, enTest) in zip(observablesExact1D, obs.pol, obs.energy)
                @test round(polTest, digits=5) ≈ polExact
                @test enTest ≈ -8.
            end
        end

        @testset "2D" begin
            
            L = (2,2)
            ind = ITensors.siteinds("S=1/2",prod(L))
            net = TTN.BinaryNetwork(L, ind)
            lat = TTN.physical_lattice(net)

            states = fill("Up", TTN.number_of_sites(net))
            ttn = TTN.ProductTreeTensorNetwork(net, states)
            ttn = TTN.increase_dim_tree_tensor_network_zeros(ttn, maxdim = 16)

            ising = TransverseFieldIsing2d(J, g, lat);
            tpo = TTN.TPO(ising, lat)
            
            obs = testObserver()
            TTN.tdvp(ttn, tpo, finaltime = tmax, timestep = dt, observer = obs)
            
            for (polExact, polTest, enTest) in zip(observablesExact2D, obs.pol, obs.energy)
                @test round(polTest, digits=5) ≈ polExact
                @test enTest ≈ -8.
            end
        end
    end
end
