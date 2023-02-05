
#=================================================================================#
#                                  ITensors                                       #
#=================================================================================#


function energy(sw::TDVPSweepHandler{N, ITensor, ITensorsBackend}) where {N}
    topPos = (number_of_layers(network(sw.ttn)), 1)
    action = ∂A(sw.pTPO, topPos)
    T = sw.ttn[topPos]
    
    return real(array(dag(T) * action(T))[1])
end;

function xPol(sw::TDVPSweepHandler{N, ITensor, ITensorsBackend}) where {N}
    return real.(vec(TTNKit.expect(sw.ttn, "X")))
end

function zPol(sw::TDVPSweepHandler{N, ITensor, ITensorsBackend}) where {N}
    return real.(vec(TTNKit.expect(sw.ttn, "Z")))
end;

function correlationMatrix(sw::TDVPSweepHandler{N, ITensor, ITensorsBackend}, op1::String, op2::String, pos::Int) where {N}
    return map(eachindex(physical_lattice(network(sw.ttn)))) do i
        return _correlation(sw.ttn, op1, op2, pos, i)
    end
end;

function entanglementEntropy(sw::TDVPSweepHandler{N, ITensor, ITensorsBackend}) where {N}
    pos1 = (number_of_layers(network(sw.ttn)), 1)
    pos2 = (number_of_layers(network(sw.ttn)) - 1, 1)
    l_ind = ITensors.commonind(sw.ttn[pos1], sw.ttn[pos2])
    (_,S,_) = ITensors.svd(sw.ttn[pos1], l_ind)
    α_sq = LinearAlgebra.diag(matrix(S)) .^ 2
    entropy = mapreduce(+, α_sq) do ev
        return -ev * log(ev)
    end
    return entropy
end;

function energyVariance(sw::TDVPSweepHandler{N, ITensor, ITensorsBackend}) where {N}
    # working in principle, but taking a veeeery long time to compute
    return 0
    
    ### not impolemented yet for TPOs
    if sw.pTPO isa ProjTPO
        return 0
    end

    tpo_data = sw.pTPO.tpo.data.data
    ttn_data = reduce(vcat,sw.ttn.data)
    tensorList = vcat(tpo_data, prime.(tpo_data), ttn_data, setprime.(ttn_data, 2))
    
    opt_seq = ITensors.optimal_contraction_sequence(tensorList)
    return real(array(contract(tensorList; sequence = opt_seq))[1])
end;

# not working yet 
function projectionErrorTest(sw::TDVPSweepHandler{N, ITensor, TTNKit.ITensorsBackend}) where {N}
    ttn = sw.ttn
    pTPO = sw.pTPO
    path = sw.path
    net = TTNKit.network(ttn)

    error = 0
    enVariance = 0

    for (pos, nextpos) in zip(reverse(path)[1:(end - 1)], reverse(path)[2:end])
        Δ = nextpos .- pos
        if Δ[1] == -1
            # orthogonalize to child
            n_child = TTNKit.index_of_child(net, nextpos)
            TTNKit._orthogonalize_to_child!(ttn, pos, n_child)

            #update environments
            pTPO = TTNKit.update_environments!(pTPO, ttn[pos], pos, nextpos)

            ttn.ortho_center[1] = nextpos[1]
            ttn.ortho_center[2] = nextpos[2]
        else
            @assert Δ[1] == 1

            action = TTNKit.∂A(pTPO, pos)
            T_error = action(ttn[pos])

            # QR-decompose time evolved tensor at pos
            idx_r = commonind(ttn[pos], ttn[nextpos])
            idx_pos = uniqueinds(ttn[pos], idx_r)
            idx_nextpos = uniqueinds(ttn[nextpos], idx_r)
            Tn, R = factorize(ttn[pos], idx_pos; tags = tags(idx_r))
            F = full_qr(ttn[pos], idx_pos)
            # G = full_qr(ttn[nextpos], idx_nextpos)
            ttn[pos] = Tn

            action2 = TTNKit.∂A2(pTPO, ttn[pos], pos)
            R_proj = action2(R)

            # multiply new R tensor onto tensor at nextpos
            ttn[nextpos] = R * ttn[nextpos]


            # move orthocenter (just for consistency), update ttnc and environments
            ttn.ortho_center[1] = nextpos[1]
            ttn.ortho_center[2] = nextpos[2]

            pTPO = TTNKit.update_environments!(pTPO, ttn[pos], pos, nextpos)
            error += real(array(dag(T_error) * T_error)[1]) - real(array(dag(R_proj) * R_proj)[1])
            if F !== nothing 
                enVariance += real(array(dag(T_error*F) * T_error*F)[1])# + real(array(dag(T2_proj) * T2_proj)[1])
            end
        end
    end
    topPos = (TTNKit.number_of_layers(net), 1)
    action = TTNKit.∂A(pTPO, topPos)
    T_proj = action(ttn[topPos])
    error += real(array(dag(T_proj) * T_proj)[1])

    return enVariance-error
end;

# not working
function projectionError(sw::TDVPSweepHandler{N, ITensor, TTNKit.ITensorsBackend}) where {N}
    ttn = sw.ttn
    pTPO = sw.pTPO
    path = sw.path
    net = TTNKit.network(ttn)

    error = 0

    for (pos, nextpos) in zip(reverse(path)[1:(end - 1)], reverse(path)[2:end])
        Δ = nextpos .- pos
        if Δ[1] == -1
            # orthogonalize to child
            n_child = TTNKit.index_of_child(net, nextpos)
            TTNKit._orthogonalize_to_child!(ttn, pos, n_child)

            #update environments
            pTPO = TTNKit.update_environments!(pTPO, ttn[pos], pos, nextpos)

            ttn.ortho_center[1] = nextpos[1]
            ttn.ortho_center[2] = nextpos[2]
        else
            @assert Δ[1] == 1

            action = TTNKit.∂A(pTPO, pos)
            T_proj = action(ttn[pos])

            # QR-decompose time evolved tensor at pos
            idx_r = commonind(ttn[pos], ttn[nextpos])
            idx_l = uniqueinds(ttn[pos], idx_r)
            Tn, R = factorize(ttn[pos], idx_l; tags = tags(idx_r))
            ttn[pos] = Tn

            action2 = TTNKit.∂A2(pTPO, ttn[pos], pos)
            R_proj = action2(R)

            # multiply new R tensor onto tensor at nextpos
            ttn[nextpos] = R * ttn[nextpos]

            # move orthocenter (just for consistency), update ttnc and environments
            ttn.ortho_center[1] = nextpos[1]
            ttn.ortho_center[2] = nextpos[2]

            pTPO = TTNKit.update_environments!(pTPO, ttn[pos], pos, nextpos)
            error += real(array(dag(T_proj) * T_proj)[1] - array(dag(R_proj) * R_proj)[1])
        end
    end
    topPos = (TTNKit.number_of_layers(net), 1)
    action = TTNKit.∂A(pTPO, topPos)
    T_proj = action(ttn[topPos])
    error += real(array(dag(T_proj) * T_proj)[1])

    return error
end;

#=================================================================================#
#                                  TensorKit                                      #
#=================================================================================#

function energy(sw::TDVPSweepHandler{N, TensorMap, TensorKitBackend}) where {N}
    topPos = (number_of_layers(network(sw.ttn)), 1)
    action = ∂A(sw.pTPO, topPos)
    T = sw.ttn[topPos]
    
    return real(array(adjoint(T) * action(T))[1])
end;

function xPol(sw::TDVPSweepHandler{N, TensorMap, TensorKitBackend}) where {N}
    σ_x = TensorMap([0 1; 1 0], ℂ^2 ← ℂ^2)
    return real.(TTNKit.expect(sw.ttn, σ_x))
end

function zPol(sw::TDVPSweepHandler{N, TensorMap, TensorKitBackend}) where {N}
    σ_z = TensorMap([1 0; 0 -1], ℂ^2 ← ℂ^2)
    return real.(TTNKit.expect(ttn, σ_z))
end;

function entanglementEntropy(sw::TDVPSweepHandler{N, TensorMap, TensorKitBackend}) where {N}
    pos = (TTNKit.number_of_layers(network(sw.ttn)), 1)
    (_, S, _, _) = TensorKit.tsvd(sw.ttn[pos], (1,), (2, 3))
    α_sq = LinearAlgebra.diag(convert(Array, S)) .^ 2
    entropy = mapreduce(+, α_sq) do ev
        return -ev * log(ev)
    end
    return entropy
end;

# need to correct indices at some point
function energyVariance(sw::TDVPSweepHandler{N, TensorMap, TensorKitBackend}) where {N}
    error("Calculation of the energyvariance for TensorMaps is currently not working!")
    @assert typeof(sw.pTPO) == TTNKit.ProjMPO

    tpo = sw.pTPO.tpo
    ttn = sw.ttn
    net = TTNKit.network(ttn)
    n_sites = TTNKit.number_of_sites(net)
    n_tensors = TTNKit.number_of_tensors(net) + n_sites

    tensorList = Vector{TTNKit.AbstractTensorMap}([])
    indexList = Vector{Vector{Int}}([])

    for ll in TTNKit.eachlayer(net)
        for pp in TTNKit.eachindex(net, ll)
            append!(tensorList, [copy(ttn[(ll, pp)]), adjoint(copy(ttn[(ll, pp)]))])
            append!(indexList, [TTNKit.internal_index_of_legs(net, (ll, pp)), TTNKit.internal_index_of_legs(net, (ll, pp))[vcat(end, 1:(end - 1))] .+ n_tensors])
        end
    end

    for pp in TTNKit.eachindex(net, 0)
        append!(tensorList, [tpo.data[tpo.mapping[pp]], tpo.data[tpo.mapping[pp]]])
        virt_leg = tpo.mapping

        append!(
            indexList,
            [[-virt_leg[pp], -virt_leg[pp] - 2 * n_sites - 2, pp, -virt_leg[pp] - 1], [-virt_leg[pp] - n_sites - 1, pp + n_tensors, -virt_leg[pp] - 2 * n_sites - 2, -virt_leg[pp] - n_sites - 2]],
        )
    end

    codom = codomain(tpo.data[1])[1]
    dom = domain(tpo.data[tpo.mapping[TTNKit.number_of_sites(net)]])[2]
    ctl = Tensor([jj == 1 ? 1 : 0 for jj in 1:dim(dom)], dom')
    ctr = Tensor([jj == dim(codom) ? 1 : 0 for jj in 1:dim(codom)], codom)

    append!(tensorList, [ctl, ctr, ctl, ctr])
    append!(indexList, [[-1], [-TTNKit.number_of_sites(net) - 1], [-TTNKit.number_of_sites(net) - 2], [-2 * TTNKit.number_of_sites(net) - 2]])

    (_, error) = TTNKit.contract_tensors(tensorList, indexList)

    return real(error[1][1])
end;

function projectionError(sw::TDVPSweepHandler{N, TensorMap, TTNKit.TensorKitBackend}) where {N}
    ttn = sw.ttn
    pTPO = sw.pTPO
    net = TTNKit.network(ttn)
    error = 0

    path = sw.path
    for (pos, nextpos) in zip(reverse(path)[1:(end - 1)], reverse(path)[2:end])
        Δ = nextpos .- pos
        if Δ[1] == -1

            # QR decomposition in direction of the sw-step
            idx_chd = TTNKit.index_of_child(net, nextpos)
            idx_dom, idx_codom = TTNKit.split_index(net, pos, idx_chd)
            perm = vcat(idx_dom..., idx_codom...)
            L, Qn = rightorth(ttn[pos], idx_dom, idx_codom)
            Qn = TensorKit.permute(Qn, Tuple(perm[1:(end - 1)]), (perm[end],))

            ttn[pos] = Qn
            ttn[nextpos] = ttn[nextpos] * L

            pTPO = TTNKit.update_environments!(pTPO, ttn[pos], pos, nextpos)
        else
            @assert Δ[1] == 1

            action = TTNKit.∂A(pTPO, pos)
            T = TensorKit.permute(action(ttn[pos]), (1, 2, 3)) #more general

            # QR-decompose time evolved tensor at pos
            Qn, R = leftorth(ttn[pos])
            ttn[pos] = Qn

            # reverse time evolution for R tensor between pos and nextpos
            action2 = TTNKit.∂A2(pTPO, ttn[pos], pos)
            Rn = TensorKit.permute(action2(R), (1, 2))

            # multiply new R tensor onto tensor at nextpos
            idx = TTNKit.index_of_child(net, pos)
            idx_dom, idx_codom = TTNKit.split_index(net, nextpos, idx)
            perm = vcat(idx_dom..., idx_codom...)
            nextT = TensorKit.permute(ttn[nextpos], idx_dom, idx_codom)
            ttn[nextpos] = TensorKit.permute(R * nextT, Tuple(perm[1:(end - 1)]), (perm[end],))

            # move orthocenter (just for consistency), update ttnc and environments
            ttn.ortho_center[1] = nextpos[1]
            ttn.ortho_center[2] = nextpos[2]

            pTPO = TTNKit.update_environments!(pTPO, ttn[pos], pos, nextpos)
            error += (adjoint(T) * T)[1] - (adjoint(Rn) * Rn)[1]
        end
    end
    topPos = (TTNKit.number_of_layers(net), 1)
    action = TTNKit.∂A(pTPO, topPos)
    T = TensorKit.permute(action(ttn[topPos]), Tuple(collect(1:number_of_child_nodes(net, topPos))))
    error += (adjoint(T) * T)[1]

    return error
end;
