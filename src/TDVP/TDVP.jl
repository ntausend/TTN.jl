using TTNKit, TensorKit, LinearAlgebra, Plots, KrylovKit
KrylovKit.:apply(A::AbstractTensorMap, x::AbstractTensorMap) = A*x


function number_of_domain_indices(t::AbstractTensorMap)
    return length(dims(domain(t)))
end

function number_of_codomain_indices(t::AbstractTensorMap)
    return length(dims(codomain(t)))
end

function number_of_indices(t::AbstractTensorMap)
    return number_of_codomain_indices(t) + number_of_domain_indices(t)
end


"""

Input: network. Returns: path for TDVP-steps as a Vector of positions

"""

function tdvp_path(net::AbstractNetwork)
    path = Vector{Tuple{Int,Int}}([(TTNKit.number_of_layers(net), 1)])

    function gotochild(pos::Tuple{Int,Int})      
        for chd_nd in TTNKit.child_nodes(net, pos)
            if chd_nd[1] > 0
                append!(path, [chd_nd])
                gotochild(chd_nd)
                append!(path, [pos])
            end
        end
    end 

    gotochild(path[1])

    return path
end

"""
    exponentiate(H::AbstractTensorMap, dt::Float64, v0::AbstractTensorMap)

Wrapper for the exponentiate function provided by the KrylovKit package

## Arguments:

---

- H::AbstractTensorMap: effective Hamiltonian function as a TensorMap, 
    indices have to be permuted s.t. H:V→V is an endomorphism, with v0 ∈ V. 
- dt::Float64: timestep
- v0::AbstractTensorMap: initial vector for algorithm,
    indices have to be permuted s.t. v0 ∈ V, see desciption of H

## Returns:

---

time evolved vector v_t = exp(-im*dt*H) * v0

"""

function exponentiate(H::AbstractTensorMap, dt::Float64, v0::AbstractTensorMap)
    result, info = KrylovKit.exponentiate(H, -1im*dt, v0)
    
    if info.converged == 0
        error("Lanczos algorithm did not converge! ", info)
    end

    return result
end


function energyvariance(ttn::TreeTensorNetwork, tpo::AbstractTensorProductOperator)
    net = TTNKit.network(ttn)
    n_sites = TTNKit.number_of_sites(net)
    n_tensors = TTNKit.number_of_tensors(net) + n_sites   

    tensorList = Vector{AbstractTensorMap}([])
    indexList =  Vector{Vector{Int}}([])

    for ll in TTNKit.eachlayer(net)
        for pp in TTNKit.eachindex(net,ll)
            append!(tensorList, [copy(ttn[(ll,pp)]),adjoint(copy(ttn[(ll,pp)]))])
            append!(indexList, [internal_index_of_legs(net,(ll,pp)),internal_index_of_legs(net,(ll,pp))[vcat(end,1:end-1)].+n_tensors])
        end
    end

    for pp in TTNKit.eachindex(net,0)
        append!(tensorList, [tpo.data[pp],tpo.data[pp]])
        append!(indexList, [[pp,-pp-2*n_sites-2,-pp,-pp-1],[-pp-2*n_sites-2,pp+n_tensors,-pp-n_sites-1,-pp-n_sites-2]])
    end

    
    dim = dims(codomain(tpo.data[1]))[end]
    append!(tensorList, [Tensor(vcat(zeros(dim-1), 1), (ℂ^dim)'), Tensor(vcat(1, zeros(dim-1)), ℂ^dim),Tensor(vcat(zeros(dim-1), 1), (ℂ^dim)'), Tensor(vcat(1, zeros(dim-1)), ℂ^dim)])
    append!(indexList, [[-1],[-TTNKit.number_of_sites(net)-1],[-TTNKit.number_of_sites(net)-2],[-2*TTNKit.number_of_sites(net)-2]])

    (nothing, error) = contract_tensors(tensorList,indexList)

    return error[1][1]
end


function tdvpforward(ttn::TreeTensorNetwork, ptpo::ProjTensorProductOperator, pos_initial::Tuple{Int,Int}, pos_final::Tuple{Int,Int}, dt::Float64)
    ttnc = copy(ttn)
    ptpoc = copy(ptpo)
    
    net = TTNKit.network(ttnc)
    Δ = pos_final .- pos_initial

    if Δ[1] == -1

        # orthogonalize to child
        n_child = TTNKit.index_of_child(net, pos_final)
        TTNKit._orthogonalize_to_child!(ttnc, pos_initial, n_child)

        ttnc.ortho_center[1] = pos_final[1]
        ttnc.ortho_center[2] = pos_final[2]

        #update environments
        update_environment(ptpoc, ttnc[pos_initial], pos_initial, pos_final)
        return (ttnc, ptpoc)

    elseif Δ[1] == 1

        # time evolve tensor at pos_initial
        old_Q_initial = TensorKit.permute(copy(ttnc[pos_initial]), Tuple(1:number_of_indices(ttnc[pos_initial])), ())
        eff_Hamiltonian = environment(ptpoc, pos_initial)
        new_Q_initial = TensorKit.permute(exponentiate(eff_Hamiltonian, dt, old_Q_initial), Tuple(1:number_of_indices(old_Q_initial)-1), (number_of_indices(old_Q_initial),))

        # QR-decomposition of time evolved tensor at pos_initial
        new_Q_initial, old_R = leftorth(new_Q_initial)
        
        # reverse time evolution for R tensor between pos_initial and pos_final
        old_R = TensorKit.permute(old_R, (1,2))
        identity = TensorKit.id((domain(new_Q_initial))')
        eff_Hamiltonian = (adjoint(new_Q_initial) ⊗ identity)*eff_Hamiltonian*(new_Q_initial ⊗ identity)
        
        new_R = TensorKit.permute(exponentiate(eff_Hamiltonian, -dt, old_R), (1,), (2,))

        # multiply new R tensor on tensor at pos_final
        idx = TTNKit.index_of_child(net, pos_initial)
        idx_dom, idx_codom = TTNKit.split_index(net, pos_final, idx)
        perm = vcat(idx_dom..., idx_codom...)
        new_Q_final = TensorKit.permute(new_R*TensorKit.permute(ttnc[pos_final], idx_dom, idx_codom), Tuple(perm[1:end-1]), (perm[end],))

        # set new tensors
        ttnc[pos_initial] = new_Q_initial
        ttnc[pos_final] = new_Q_final

        # move orthocenter (just for consistency), update ttnc and environments  
        ttnc.ortho_center[1] = pos_final[1]
        ttnc.ortho_center[2] = pos_final[2]

        update_environment(ptpoc, new_Q_initial, pos_initial, pos_final)
        return (ttnc, ptpoc)

    else
        error("Invalid direction for TDVP step: ", pos_initial, ", ", pos_final)
    end
end


function tdvpbackward(ttn::TreeTensorNetwork, ptpo::ProjTensorProductOperator, pos_initial::Tuple{Int,Int}, pos_final::Tuple{Int,Int}, dt::Float64)
    ttnc = copy(ttn)
    ptpoc = copy(ptpo)

    net = TTNKit.network(ttnc)
    Δ = pos_final .- pos_initial

    if Δ[1] == 1
        
        # orthogonalize to parent
        TTNKit._orthogonalize_to_parent!(ttnc, pos_initial)
        ttnc.ortho_center[1] = pos_final[1]
        ttnc.ortho_center[2] = pos_final[2]

        #update environments
        update_environment(ptpoc, ttnc[pos_initial], pos_initial, pos_final)
        return (ttnc, ptpoc)

    elseif Δ[1] == -1

        # QR decomposition in direction of the TDVP-step
        idx_chd = TTNKit.index_of_child(net, pos_final)
        idx_dom, idx_codom = TTNKit.split_index(net, pos_initial, idx_chd)
        perm = vcat(idx_dom..., idx_codom...)
        old_L, new_Q_initial = rightorth(ttnc[pos_initial], idx_dom, idx_codom)
        old_L = TensorKit.permute(old_L,(1,2))
        new_Q_initial = TensorKit.permute(new_Q_initial, Tuple(perm[1:end-1]), (perm[end],))
        ttnc[pos_initial] = new_Q_initial
        new_Q_initial = TensorKit.permute(new_Q_initial, idx_codom, idx_dom)

        # reverse time evolution for R tensor between pos_initial and pos_final
        eff_Hamiltonian = environment(ptpoc, pos_initial)
        eff_Hamiltonian = TensorKit.permute(eff_Hamiltonian, Tuple(vcat(idx_dom..., idx_codom...)), Tuple(vcat((idx_dom.+length(perm))..., (idx_codom.+length(perm))...)))
        identity = TensorKit.id((domain(new_Q_initial))')
        eff_Hamiltonian = (identity ⊗ adjoint(new_Q_initial))*eff_Hamiltonian*(identity ⊗ new_Q_initial)
        new_L = TensorKit.permute(exponentiate(eff_Hamiltonian, -dt, old_L), (1,), (2,))

        # multiply new R tensor on tensor at pos_final
        old_Q_final = TensorKit.permute(ttnc[pos_final]*new_L, Tuple(1:number_of_indices(ttnc[pos_final])))

        # update environments time evolve tensor at pos_final
        update_environment(ptpoc, ttnc[pos_initial], pos_initial, pos_final)
        eff_Hamiltonian = environment(ptpoc, pos_final)
        new_Q_final = TensorKit.permute(exponentiate(eff_Hamiltonian, dt, old_Q_final), Tuple(1:number_of_indices(old_Q_final)-1), (number_of_indices(old_Q_final),))

        # set new tensor and move orthocenter (just for consistency)
        ttnc[pos_final] = new_Q_final
        ttnc.ortho_center[1] = pos_final[1]
        ttnc.ortho_center[2] = pos_final[2]

        return (ttnc, ptpoc)
    else
        error("Invalid direction for TDVP step", pos_initial, ", ", pos_final)
    end
end


function tdvptopnode(ttn::TreeTensorNetwork, ptpo::ProjTensorProductOperator, dt::Float64)
    ttnc = copy(ttn)
    ptpoc = copy(ptpo)
    net = TTNKit.network(ttnc)

    pos = (TTNKit.number_of_layers(net), 1) 
    old_Q = TensorKit.permute(copy(ttnc[pos]), Tuple(1:number_of_indices(ttnc[pos])))

    # time evolution of top-node
    eff_Hamiltonian = environment(ptpoc, pos)
    new_Q = TensorKit.permute(exponentiate(eff_Hamiltonian, dt, old_Q), Tuple(1:number_of_indices(old_Q)-1), (number_of_indices(old_Q),))
    ttnc[pos] = new_Q
    
    return (ttnc, ptpoc)
end


# observables
function x_pol(ttn::TreeTensorNetwork)
    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)
    σ_x = TensorMap([0 1; 1 0], ℂ^2 ← ℂ^2)   
    
    return sum(TTNKit.expect(ttn, σ_x))/len
end

function xx_pol(ttn::TreeTensorNetwork, distance::Integer)
    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)
    σ_x = TensorMap([0 1; 1 0], ℂ^2 ← ℂ^2)   

    correlations = [TTNKit.correlation(ttn, σ_x, σ_x, pp, pp+distance) for pp in 1:len-distance]
    
    return sum(correlations)/len
end

function z_pol(ttn::TreeTensorNetwork)
    len = TTNKit.number_of_sites(TTNKit.network(ttn))
    σ_z = TensorMap([1 0; 0 -1], ℂ^2 ← ℂ^2) 
    
    return sum(TTNKit.expect(ttn, σ_z))/len
end

function zz_pol(ttn::TreeTensorNetwork, distance::Integer)
    net = TTNKit.network(ttn)
    len = TTNKit.number_of_sites(net)
    σ_z = TensorMap([1 0; 0 -1], ℂ^2 ← ℂ^2) 

    correlations = [TTNKit.correlation(ttn, σ_z, σ_z, pp, pp+distance) for pp in 1:len-distance]
    
    return sum(correlations)/len
end

function entanglementEntropy(ttn::TreeTensorNetwork)
    net = TTNKit.network(ttn)
    pos = (TTNKit.number_of_layers(net),1)
    U, S, V, eps = TensorKit.tsvd(ttn[pos], (1,), (2,3))
    α_sq = LinearAlgebra.diag(convert(Array, S)).^2
    entropy = mapreduce(+, α_sq) do ev
        return -ev*log(ev)
    end
    return entropy 
end


function tdvprun(ttn::TreeTensorNetwork, ptpo::ProjTensorProductOperator, dt::Float64, tmax::Float64)
    observables = Any[]
    path = tdvp_path(TTNKit.network(ttn))
    t = 0

    while t < tmax
        println("Current time: $(t)")

        for  (pos_initial, pos_final) in zip(path[1:end-1], path[2:end])
            (ttn, ptpo) = tdvpforward(ttn, ptpo, pos_initial, pos_final, dt/2)
        end
        
        (ttn, ptpo) = tdvptopnode(ttn, ptpo, dt)

        for  (pos_initial, pos_final) in zip(reverse(path)[1:end-1], reverse(path)[2:end])
            (ttn, ptpo) = tdvpbackward(ttn, ptpo, pos_initial, pos_final, dt/2)
        end


        Q = TensorKit.permute(ttn[path[1]], (1,2,3), ())
        energy = real((adjoint(Q) * environment(ptpo, path[1]) * Q)[1][1])

        append!(observables, [[t, energy, real(x_pol(ttn)), real(z_pol(ttn)), entanglementEntropy(ttn)]])
        
        t += dt
    end
    
    p1 = plot([ob[1] for ob in observables], [ob[2] for ob in observables], label = "energy")
    p2 = plot([ob[1] for ob in observables], [ob[3] for ob in observables], label = "x-pol")
    p3 = plot([ob[1] for ob in observables], [ob[4] for ob in observables], label = "z-pol")
    p4 = plot([ob[1] for ob in observables], [ob[5] for ob in observables], label = "entanglement entropy")
    
    plot(p1, p2, p3, p4, layout = 4)
end

# --------------- To Do ---------------
# 
# implement some test functions for environment etc
#
# error measure? 
#
