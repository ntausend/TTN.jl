using TTNKit, TensorKit, LinearAlgebra, Plots, KrylovKit

KrylovKit.:apply(A::AbstractTensorMap, x::AbstractTensorMap) = A*x

n_layers = 3
net = BinaryChainNetwork(n_layers, TTNKit.SpinHalfNode)
len = TTNKit.number_of_sites(net)
expected = ones(len) #rand(0:1, 2^n_layers) #
states = map(expected) do s
    return  "Right" #s== 0 ? "Down" : "Up" #
end


ttn = ProductTreeTensorNetwork(net, states) 
# ttn = increase_dim_tree_tensor_network_zeros(ttn, maxdim = 10)
ttn = increase_dim_tree_tensor_network_randn(ttn, maxdim = 10, factor = 10e-12)

function x_pol(ttn::TreeTensorNetwork)
    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)
    σ_x = TensorMap([0 1; 1 0], ℂ^2 ← ℂ^2)   
    
    return sum(TTNKit.expect(ttn, σ_x))/len
end

function z_pol(ttn::TreeTensorNetwork)
    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(TTNKit.network(ttn))
    σ_z = TensorMap([1 0; 0 -1], ℂ^2 ← ℂ^2) 
    
    return sum(TTNKit.expect(ttn, σ_z))/len
end

function number_of_domain_indices(t::AbstractTensorMap)
    return length(dims(domain(t)))
end

function number_of_codomain_indices(t::AbstractTensorMap)
    return length(dims(codomain(t)))
end

function number_of_indices(t::AbstractTensorMap)
    return number_of_codomain_indices(t) + number_of_domain_indices(t)
end

function TDVP_path(ttn::TreeTensorNetwork)
    net = TTNKit.network(ttn)
    path = Vector{Tuple{Int,Int}}([(TTNKit.number_of_layers(net), 1)])

    function go_to_child(pos::Tuple{Int,Int})      
        for chd_nd in TTNKit.child_nodes(net, pos)
            if chd_nd[1] > 0
                append!(path, [chd_nd])
                go_to_child(chd_nd)
                append!(path, [pos])
            end
        end
    end 

    go_to_child(path[1])

    return path
end


"TDVP step in the forward loop; time evolution is propagated from bottom to top for half a time step"

function TDVP_step_forward_loop(ttn::TreeTensorNetwork, ptpo::ProjTensorProductOperator, pos_initial::Tuple{Int,Int}, pos_final::Tuple{Int,Int}, dt::Float64)
    net = TTNKit.network(ttn)
    Δ = pos_final .- pos_initial

    if Δ[1] == -1

        # orthogonalize to child
        n_child = TTNKit.index_of_child(net, pos_final)
        TTNKit._orthogonalize_to_child!(ttn, pos_initial, n_child)

        ttn.ortho_center[1] = pos_final[1]
        ttn.ortho_center[2] = pos_final[2]

        #update environments
        update_environment(ptpo, ttn[pos_initial], pos_initial, pos_final)

        return ttn

    elseif Δ[1] == 1

        # time evolve tensor at pos_initial
        old_Q_initial = TensorKit.permute(copy(ttn[(pos_initial)]), Tuple(1:number_of_indices(ttn[(pos_initial)])), ())
        eff_Hamiltonian = environment(ptpo, pos_initial)
        new_Q_initial, info = KrylovKit.exponentiate(eff_Hamiltonian, -1im*dt, old_Q_initial)
        new_Q_initial = TensorKit.permute(new_Q_initial, Tuple(1:number_of_indices(new_Q_initial)-1), (number_of_indices(new_Q_initial),))

        if info.converged == 0
            println(info)
            error("Lanczos did not converge")
        end

        # QR-decomposition of time evolved tensor at pos_initial
        new_Q_initial, old_R = leftorth(new_Q_initial)
        
        # reverse time evolution for R tensor between pos_initial and pos_final
        old_R = TensorKit.permute(old_R, (1,2))
        identity = TensorKit.id((domain(new_Q_initial))')
        eff_Hamiltonian = (adjoint(new_Q_initial) ⊗ identity)*eff_Hamiltonian*(new_Q_initial ⊗ identity)
        
        new_R, info = KrylovKit.exponentiate(eff_Hamiltonian, 1im*dt, old_R)
        new_R = TensorKit.permute(new_R, (1,), (2,))

        if info.converged == 0
            println(info)
            error("Lanczos did not converge")
        end

        # multiply new R tensor on tensor at pos_final
        idx = TTNKit.index_of_child(net, pos_initial)
        idx_dom, idx_codom = TTNKit.split_index(net, pos_final, idx)
        perm = vcat(idx_dom..., idx_codom...)
        new_Q_final = TensorKit.permute(new_R*TensorKit.permute(ttn[pos_final], idx_dom, idx_codom), Tuple(perm[1:end-1]), (perm[end],))

        # set new tensors
        ttn[pos_initial] = new_Q_initial
        ttn[pos_final] = new_Q_final

        # update environments
        update_environment(ptpo, new_Q_initial, pos_initial, pos_final)

        ttn.ortho_center[1] = pos_final[1]
        ttn.ortho_center[2] = pos_final[2]

        return ttn 

    else
        error("Invalid direction for TDVP step: ", pos_initial, ", ", pos_final)
    end
end

"TDVP step in the backward loop; time evolution is propagated from top to bottom for half a time step"

function TDVP_step_backward_loop(ttn::TreeTensorNetwork, ptpo::ProjTensorProductOperator, pos_initial::Tuple{Int,Int}, pos_final::Tuple{Int,Int}, dt::Float64)
    net = TTNKit.network(ttn)
    Δ = pos_final .- pos_initial

    if Δ[1] == 1
        
        # orthogonalize to parent
        TTNKit._orthogonalize_to_parent!(ttn, pos_initial)
        ttn.ortho_center[1] = pos_final[1]
        ttn.ortho_center[2] = pos_final[2]

        #update environments
        update_environment(ptpo, ttn[pos_initial], pos_initial, pos_final)

        return ttn

    elseif Δ[1] == -1

        idx_chd = TTNKit.index_of_child(net, pos_final)
        idx_dom, idx_codom = TTNKit.split_index(net, pos_initial, idx_chd)
        perm = vcat(idx_dom..., idx_codom...)
        old_L, new_Q_initial = rightorth(ttn[pos_initial], idx_dom, idx_codom)
        old_L = TensorKit.permute(old_L,(1,2))
        new_Q_initial = TensorKit.permute(new_Q_initial, Tuple(perm[1:end-1]), (perm[end],))
        ttn[pos_initial] = new_Q_initial
        new_Q_initial = TensorKit.permute(new_Q_initial, idx_codom, idx_dom)

        eff_Hamiltonian = environment(ptpo, pos_initial)
        eff_Hamiltonian = TensorKit.permute(eff_Hamiltonian, Tuple(vcat(idx_dom..., idx_codom...)), Tuple(vcat((idx_dom.+length(perm))..., (idx_codom.+length(perm))...)))

        identity = TensorKit.id((domain(new_Q_initial))')

        eff_Hamiltonian = (identity ⊗ adjoint(new_Q_initial))*eff_Hamiltonian*(identity ⊗ new_Q_initial)

        new_L, info = KrylovKit.exponentiate(eff_Hamiltonian, 1im*dt, old_L)
        new_L = TensorKit.permute(new_L, (1,), (2,))

        if info.converged == 0
            println(info)
            error("Lanczos did not converge")
        end

        old_Q_final = TensorKit.permute(ttn[pos_final]*new_L, Tuple(1:number_of_indices(ttn[pos_final])))

        update_environment(ptpo, ttn[pos_initial], pos_initial, pos_final)
        eff_Hamiltonian = environment(ptpo, pos_final)

        new_Q_final, info = KrylovKit.exponentiate(eff_Hamiltonian, -1im*dt, old_Q_final)
        new_Q_final = TensorKit.permute(new_Q_final, Tuple(1:number_of_indices(new_Q_final)-1), (number_of_indices(new_Q_final),))

        if info.converged == 0
            println(info)
            error("Lanczos did not converge")
        end

        ttn[pos_final] = new_Q_final

        ttn.ortho_center[1] = pos_final[1]
        ttn.ortho_center[2] = pos_final[2]

        return ttn 

    else
        error("Invalid direction for TDVP step", pos_initial, ", ", pos_final)
    end
end

function TDVP_step_topNode(ttn::TreeTensorNetwork, ptpo::ProjTensorProductOperator, dt::Float64)
 
    net = TTNKit.network(ttn)
    pos = (TTNKit.number_of_layers(net), 1)
    old_Q = TensorKit.permute(copy(ttn[pos]), Tuple(1:number_of_indices(ttn[pos])))

    eff_Hamiltonian = environment(ptpo, pos)
    new_Q, info = KrylovKit.exponentiate(eff_Hamiltonian, -1im*dt, old_Q)
    new_Q = TensorKit.permute(new_Q, Tuple(1:number_of_indices(new_Q)-1), (number_of_indices(new_Q),))

    if info.converged == 0
        println(info)
        error("Lanczos did not converge")
    end

    ttn[pos] = new_Q
    
    return ttn 
end

function TDVP(ttn::TreeTensorNetwork)
    tpo = transverseIsingHamiltonian((-1., -2.), TTNKit.physical_lattice(TTNKit.network(ttn)))
    ptpo = ProjTensorProductOperator(ttn, tpo)
    path = TDVP_path(ttn)

    observables = Any[]

    t = 0
    timestep = 1e-2
    tmax = 2

    while t < tmax
        println("t: ", t)

        for i in 1:length(path)-1
            
            (pos_initial, pos_final) = (path[i], path[i+1])
            ttn = TDVP_step_forward_loop(ttn, ptpo, pos_initial, pos_final, timestep/2)
        end
        
        ttn = TDVP_step_topNode(ttn, ptpo, timestep)

        for i in 1:length(path)-1
            (pos_initial, pos_final) = (reverse(path)[i], reverse(path)[i+1])
            ttn = TDVP_step_backward_loop(ttn, ptpo, pos_initial, pos_final, timestep/2)
        end

        t += timestep

        Q = TensorKit.permute(ttn[path[1]], (1,2,3), ())
        energy = real((adjoint(Q) * environment(ptpo, path[1]) * Q)[1][1])
        # println("norm: ", (adjoint(Q)*Q)[1,1])

        append!(observables, [[t, energy, real(x_pol(ttn)), real(z_pol(ttn))]])
    end
    
    p1 = plot([ob[1] for ob in observables], [ob[2] for ob in observables], label = "energy")
    p2 = plot([ob[1] for ob in observables], [ob[3] for ob in observables], label = "x-pol")
    p3 = plot([ob[1] for ob in observables], [ob[4] for ob in observables], label = "z-pol")
    
    plot(p1, p2, p3, layout = 3)
end