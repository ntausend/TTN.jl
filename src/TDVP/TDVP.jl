using TTNKit, TensorKit, LinearAlgebra, Plots

n_layers = 3
net = BinaryChainNetwork(n_layers, TTNKit.SpinHalfNode)
len = TTNKit.number_of_sites(net)
expected = rand(0:1, 2^n_layers) #ones(len) #
states = map(expected) do s
    return  "Right" #s== 0 ? "Down" : "Up" #
end

ttn = ProductTreeTensorNetwork(net, states) 
ttn = increase_dim_tree_tensor_network(ttn, maxdim = 4)

function energy(ttn::TreeTensorNetwork{D}) where{D}
    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)
    t = contracted(ttn)
    isingHamiltonian = transverseIsingHamiltonian1D((-1., -2.), ℂ^2, len)
    tensorList = Any[adjoint(t), t, isingHamiltonian]
    indexList = Any[1:len, 1+len:2*len, 1:2*len]

    res = TensorKit.ncon(tensorList, indexList)

    return res
end

function x_pol(ttn::TreeTensorNetwork{D}) where{D}
    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)
    σ_x = TensorMap([0 1; 1 0], ℂ^2 ← ℂ^2)   
    
    return sum(TTNKit.expect(ttn, σ_x))/len
end

function z_pol(ttn::TreeTensorNetwork{D}) where{D}
    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)
    σ_z = TensorMap([1 0; 0 -1], ℂ^2 ← ℂ^2) 
    
    return sum(TTNKit.expect(ttn, σ_z))/len
end

function numbering(ttn::TreeTensorNetwork{D}) where{D}
    net = TTNKit.network(ttn)
    start = 1
    len = 0
    res = Any[-start:-1:-TTNKit.number_of_sites(net)]

    for ll in TTNKit.eachlayer(net)
        len += TTNKit.length(TTNKit.layer(ttn, ll))
        append!(res, [start:len])
        start += TTNKit.length(TTNKit.layer(ttn, ll))
    end
    
    return res
end

function chd_env(ttn::TreeTensorNetwork{D}, pos::Tuple{Int, Int}) where{D}
    net = TTNKit.network(ttn)

    pos[1] == 1 && ( return (TensorKit.id(codomain(ttn[pos])), TensorKit.id(codomain(ttn[pos]))) )

    chd_nd1, chd_nd2 = TTNKit.child_nodes(net, pos)
    layer = chd_nd1[1]
    tensor_chd1 = ttn[chd_nd1]
    tensor_chd2 = ttn[chd_nd2]

    chd_nd1 = [chd_nd1] 
    chd_nd2 = [chd_nd2]

    while layer > 1
        chd_nd1_temp = Vector{Tuple{Int, Int}}()
        chd_nd2_temp = Vector{Tuple{Int, Int}}()

        tensor_chd1_temp = Vector{TensorMap}()
        tensor_chd2_temp = Vector{TensorMap}()

        for n in chd_nd1
            nd1, nd2 = TTNKit.child_nodes(net, n)
            append!(chd_nd1_temp, [nd1, nd2])
            append!(tensor_chd1_temp, [ttn[nd1], ttn[nd2]])
        end

        for n in chd_nd2
            nd1, nd2 = TTNKit.child_nodes(net, n)
            append!(chd_nd2_temp, [nd1, nd2])
            append!(tensor_chd2_temp, [ttn[nd1], ttn[nd2]])
        end

        chd_nd1 = chd_nd1_temp
        chd_nd2 = chd_nd2_temp

        tensor_chd1 = reduce(⊗, tensor_chd1_temp) * tensor_chd1
        tensor_chd2 = reduce(⊗, tensor_chd2_temp) * tensor_chd2

        layer -= 1
                        
    end

    return (tensor_chd1, tensor_chd2)
end


function environment(ttn::TreeTensorNetwork{D}, pos::Tuple{Int, Int}) where{D}
    ttnc = copy(ttn)

    net = TTNKit.network(ttnc)
    number = numbering(ttnc)
    len = TTNKit.number_of_sites(net)

    chd_nds = TTNKit.child_nodes(net, pos)
    par_nd = TTNKit.parent_node(net, pos)

    tensorList = Any[]
    indexList = Any[]

    for pp in TTNKit.eachindex(net,1)
        spin1 = number[1][2*pp-1]
        spin2 = number[1][2*pp]
        parent_ind = number[2][pp]
        if (1,pp) in chd_nds
            append!(tensorList, [ttn[(1,pp)]])
            append!(indexList, [[spin1, spin2, -len-TTNKit.index_of_child(net, (1,pp))]])
        elseif (1,pp) == pos
            append!(tensorList, [TensorKit.id(codomain(ttnc[pos]))])
            append!(indexList, [[spin1, spin2, -len-1, -len-2]])
        else
            append!(tensorList, [ttn[(1,pp)]])
            append!(indexList, [[spin1, spin2, parent_ind]])
        end
    end
    for ll in Iterators.drop(TTNKit.eachlayer(net), 1)
        for pp in TTNKit.eachindex(net,ll)
            child_ind1 = number[ll][2*pp-1]
            child_ind2 = number[ll][2*pp]
            parent_ind = number[ll+1][pp]
            if (ll,pp) in chd_nds
                append!(tensorList, [ttn[(ll,pp)]])
                append!(indexList, [[child_ind1,child_ind2, -len-TTNKit.index_of_child(net, (ll,pp))]])
            elseif (ll,pp) == par_nd
                append!(tensorList, [ttn[(ll,pp)]])
                if TTNKit.index_of_child(net, pos) == 1
                    append!(indexList, [[-len-3, child_ind2, parent_ind]])
                else
                    append!(indexList, [[child_ind1, -len-3, parent_ind]])
                end
            elseif (ll,pp) != pos
                append!(tensorList, [ttn[(ll,pp)]])
                append!(indexList, [[child_ind1,child_ind2,parent_ind]])
            end
        end
    end
    if pos[1] == TTNKit.number_of_layers(net)
        append!(tensorList, [TensorKit.id(domain(ttnc[pos]))])
        append!(indexList, [[-len-3, number[TTNKit.number_of_layers(net)+1][1]]])
    end
    append!(tensorList, [Tensor([1], domain(ttnc[(TTNKit.number_of_layers(net),1)])^1)])
    append!(indexList, [[number[TTNKit.number_of_layers(net)+1][1]]])

    
    return (TensorKit.permute(TensorKit.ncon(tensorList, indexList), Tuple(1:len), Tuple(len+1:len+3)), TensorKit.permute(ttnc[pos], (1,2,3), ()))
end


function Lanczos(v::TensorMap, Hamiltonian::TensorMap, timestep::Float64) 
    
    N = dim(codomain(v))

    norm = sqrt(real((adjoint(v)*v)))
    v = v/norm

    #Lanczos-vector
    Vec = Any[v]

    w = Hamiltonian * v
    α = real((adjoint(w)*v)[1,1])
    w -= α*v

    #vectors containing the normalized α and β factors
    A = Vector{Float64}([α])
    B = Vector{Float64}([])

    for i in 2:N
        β = sqrt(real((adjoint(w)*w)[1,1]))

        #in case the procedure fails and we get a vector with zero norm, pick a random Tensor, othogonalize to the previous vectors and normalize it
        if β < 1e-8 
            w = TensorMap(randn, codomain(v) ← domain(v))

            for v in Vec
                w -= (adjoint(v)*w)[1,1] * v
            end

            w /= sqrt(real((adjoint(w)*w)[1,1]))
        else 
            w /= β
        end

        append!(B, [β])

        #reorthonormalization for a better numerical stability
        for v in Vec
            w -= (adjoint(v)*w)[1,1] * v
        end
        w /= sqrt(real((adjoint(w)*w)[1,1]))

        append!(Vec, [w])
        w = Hamiltonian * Vec[i]
        α = real((adjoint(w)*Vec[i])[1,1])
        append!(A, [α])
        w = w - A[i]*Vec[i]- B[i-1]*Vec[i-1]
    end

    adj_Vec = Any[adjoint(el) for el in Vec]

    # LAPACK routine for calculating eigenvalues and -vectors
    eval, evec = LinearAlgebra.LAPACK.stev!('V', A, B) 
    exp_eval = [exp(-1im * timestep * ev) for ev in eval]

    return sum(Vec.*((evec*Diagonal(exp_eval)*transpose(evec))*adj_Vec))
end

"TDVP step in the forward loop; time evolution is propagated from bottom to top for half a time step"

function TDVP_step_forward_loop(ttn::TreeTensorNetwork{D}, pos_initial::Tuple{Int,Int}, pos_final::Tuple{Int,Int}, dt::Float64) where{D}

    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)

    Δ = pos_final .- pos_initial

    if Δ[1] == -1
        n_child = TTNKit.index_of_child(net, pos_final)
        TTNKit._orthogonalize_to_child!(ttn, pos_initial, n_child)
        return ttn

    elseif Δ[1] != 1
        error("Invalid direction for TDVP step: ", pos_initial, ", ", pos_final)
    end
    
    env_Q, Q_old = environment(ttn, pos_initial)

    isingHamiltonian = transverseIsingHamiltonian1D((-1., -2.), ℂ^2, len)
    eff_Hamiltonian_Q = adjoint(env_Q) * isingHamiltonian * env_Q

    integrator_Q = Lanczos(Q_old, eff_Hamiltonian_Q, dt)

    ttn[pos_initial] = TensorKit.permute(integrator_Q * Q_old,(1,2),(3,))
    
    Q, R = leftorth(ttn[pos_initial])
    Q = TensorKit.permute(Q, (1,2,3))
    R = TensorKit.permute(R,(1,2))

    @tensor eff_Hamiltonian_R[(-4,-3); (-2,-1)] := eff_Hamiltonian_Q[3,4,-3,1,2,-1]*Q[1,2,-2]*adjoint(Q)[3,4,-4]

    integrator_R = Lanczos(R, eff_Hamiltonian_R, -dt)
    update_R = TensorKit.permute(integrator_R * R, (1,), (2,))

    idx = TTNKit.index_of_child(net, pos_initial)
    idx_dom, idx_codom = TTNKit.split_index(net, pos_final, idx)

    perm = vcat(idx_dom..., idx_codom...)
    res = update_R*TensorKit.permute(ttn[pos_final], idx_dom, idx_codom)
    
    res = TensorKit.permute(res, Tuple(perm[1:end-1]), (perm[end],))

    ttn[pos_initial] = TensorKit.permute(Q, (1,2), (3,))
    ttn[pos_final] = res

    return ttn 
end


"TDVP step in the backward loop; time evolution is propagated from top to bottom for half a time step"
function TDVP_step_backward_loop(ttn::TreeTensorNetwork{D}, pos_initial::Tuple{Int,Int}, pos_final::Tuple{Int,Int}, dt::Float64) where{D}

    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)

    Δ = pos_final .- pos_initial

    if Δ[1] == 1
        TTNKit._orthogonalize_to_parent!(ttn, pos_initial)
        return ttn

    elseif Δ[1] != -1
        error("Invalid direction for TDVP step", pos_initial, ", ", pos_final)
    end

    env_Q, Q_old = environment(ttn, pos_initial)

    isingHamiltonian = transverseIsingHamiltonian1D((-1., -2.), ℂ^2, len)
    eff_Hamiltonian_Q = adjoint(env_Q) * isingHamiltonian * env_Q

    n_child = TTNKit.index_of_child(net, pos_final)    
    idx_dom, idx_codom = TTNKit.split_index(net, pos_initial, n_child)

    perm = vcat(idx_dom..., idx_codom...)
    L, Q = rightorth(ttn[pos_initial], idx_dom, idx_codom)
    L = TensorKit.permute(L,(1,2))
    ttn[pos_initial] = TensorKit.permute(Q, Tuple(perm[1:end-1]), (perm[end],))   

    Q = TensorKit.permute(Q, (1,2,3))

    if n_child == 1
        @tensor eff_Hamiltonian_L[(-1,-2); (-3,-4)] := adjoint(Q)[-2,1,2]*eff_Hamiltonian_Q[-1,1,2,-3,3,4]*Q[-4,3,4]
    else
        @tensor eff_Hamiltonian_L[(-1,-2); (-3,-4)] := adjoint(Q)[1,-2,2]*eff_Hamiltonian_Q[1,-1,2,3,-3,4]*Q[3,-4,4]
    end

    integrator_L = Lanczos(L, eff_Hamiltonian_L, -dt)
    update_L = integrator_L * L

    update_L = TensorKit.permute(update_L, (1,), (2,))

    ttn[pos_final] = ttn[pos_final]*update_L
    env_Q_final, Q_final = environment(ttn, pos_final)


    eff_Hamiltonian_Q_final = adjoint(env_Q_final) * isingHamiltonian * env_Q_final

    integrator = Lanczos(Q_final, eff_Hamiltonian_Q_final, dt)

    ttn[pos_final] = TensorKit.permute(integrator * Q_final,(1,2),(3,))
    
    return ttn 
end


function TDVP_step_topNode(ttn::TreeTensorNetwork{D}, dt::Float64) where{D}
 
    net = TTNKit.network(ttn)
    pos = (TTNKit.number_of_layers(net), 1)
    env_Q, Q_old = environment(ttn, pos)

    isingHamiltonian = transverseIsingHamiltonian1D((-1., -2.), ℂ^2, len)

    eff_Hamiltonian_Q = adjoint(env_Q) * isingHamiltonian * env_Q

    integrator = Lanczos(Q_old, eff_Hamiltonian_Q, dt)

    ttn[pos] = TensorKit.permute(integrator * Q_old,(1,2),(3,))
    
    return ttn 
end

function TDVP(ttn::TreeTensorNetwork{D}) where{D}
    path = [[(2,1),(1,2)],[(1,2),(2,1)],[(2,1),(1,1)],[(1,1),(2,1)]] #[[(3,1),(2,2)],[(2,2),(1,4)],[(1,4),(2,2)],[(2,2),(1,3)],[(1,3),(2,2)],[(2,2),(3,1)],[(3,1),(2,1)],[(2,1),(1,2)],[(1,2),(2,1)],[(2,1),(1,1)],[(1,1),(2,1)],[(2,1),(3,1)]] #
    timestep = 1e-3

    ttnc = copy(ttn)
    TTNKit.move_ortho!(ttnc, (2,1))

    en = Any[]
    x = Any[]
    z = Any[]
    time = Any[]

    t = 0
    tmax = 5

    while t < tmax
        println("t: ", t)
        for pos in path
            pos_initial, pos_final = pos
            ttnc = TDVP_step_forward_loop(ttnc, pos_initial, pos_final, timestep/2)
        end

        ttnc = TDVP_step_topNode(ttnc, timestep)
        
        for pos in reverse(path)
            pos_final, pos_initial = pos
            ttnc = TDVP_step_backward_loop(ttnc, pos_initial, pos_final, timestep/2)
        end

        t += timestep

        append!(time, [t])
        append!(en, [real(energy(ttnc)[1,1])])
        append!(x, [real(x_pol(ttnc))])
        append!(z, [real(z_pol(ttnc))])
    end
    
    p1 = plot(time, en)
    p2 = plot(time, x)
    p3 = plot(time, z)
    plot(p1, p2, p3, layout = 3)
end

TDVP(ttn)