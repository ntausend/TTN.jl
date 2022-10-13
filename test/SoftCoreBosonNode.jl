using TTNKit, TensorKit
using Test


elT = Float64

function test_scb(n_bosons, conserve_n, conserve_pa, parity)
    states = map(0:n_bosons) do n
        [jj == n  ? 1 : 0 for jj in 0:n_bosons]
    end
    state_names = map(jj -> "$jj", 0:n_bosons)
    if(conserve_n)
        S = GradedSpace{U1Irrep, TensorKit.SortedVectorDict{U1Irrep, Int64}}
        I = U1Irrep
        hilbsp = U1Space([jj => 1 for jj in 0:n_bosons])
        domain_fn = state_name -> U1Space(parse(Int64, state_name) => 1)
        st_fn = st -> st
        spaces = [jj => 1 for jj in 0:n_bosons]
    elseif(conserve_pa)
        S = GradedSpace{ZNIrrep{parity}, NTuple{parity,Int64}}        
        I = ZNIrrep{parity}
        psbl_chrg = 0:parity-1
        chrgs = [mod(jj, parity) for jj in 0:n_bosons]
        sectors_degen = map(psbl_chrg) do c
            c => sum(chrgs .== c)
        end
        hilbsp = ZNSpace{parity}(sectors_degen)
        domain_fn = state_name -> ZNSpace{parity}(mod(parse(Int64, state_name), parity) => 1)

        function st_fn_parity(st)
            chrg_raw = findfirst(!iszero, st) - 1
            chrg = mod(chrg_raw, parity)
            chrg_pref = mapreduce(+, 1:chrg, init = 0) do jj
                    last(sectors_degen[jj])
            end
        
            index_shift = div(chrg_raw, parity)
            new_index = index_shift + chrg_pref + 1
            return [jj == new_index ? 1 : 0 for jj in 1:n_bosons+1]
        end
        st_fn = st -> st_fn_parity(st)
        spaces = [mod(jj,parity) => 1 for jj in 0:n_bosons]
    else
        S = ComplexSpace
        I = Trivial
        hilbsp = S(n_bosons+1)
        domain_fn = _ -> S(1)
        st_fn = st -> st
        spaces = n_bosons+1 
    end


    @testset "conserve_n = $conserve_n, conserve_parity = $conserve_pa" begin
        nd_scb = TTNKit.SoftCoreBosonNode(1, "1"; number_of_bosons = n_bosons, conserve_qns = conserve_n, conserve_parity = conserve_pa, parity = parity)

        # properties
        @test TTNKit.position(nd_scb) == 1
        @test TTNKit.description(nd_scb) == "SCB 1"
        @test sectortype(nd_scb) == I
        @test spacetype(nd_scb) <: S

        # creating virtual node copies testing
        nd_2 = TTNKit.nodetype(nd_scb)
        @test nd_2 == Node{S, I}
        @test !(nd_2 == nd_scb)

        foreach(zip(state_names, states)) do (st_name, st)
            @test TTNKit.state(nd_scb, Val(Symbol(st_name))) == st
            st_o = TTNKit.state(nd_scb, st_name, elT)
            @test st_o == TensorMap(st_fn(st), hilbsp, domain_fn(st_name))
        end
        @test TTNKit.spaces(nd_scb) == spaces
    end
end

@testset "SoftCoreBoson, n_bosons=$n" for n in 1:6
    test_scb(n, false, false, 2)
    test_scb(n, true, false, 2)
    @testset "Parity: $p" for p in 2:4
        test_scb(n, false, true, p)
    end
end