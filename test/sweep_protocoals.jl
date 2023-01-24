
using TensorKit, TTNKit
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
        #### TODO ####         
    end
end