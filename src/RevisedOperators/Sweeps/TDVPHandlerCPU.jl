mutable struct TDVPSweepHandlerCPU{N<:AbstractNetwork,T} <: AbstractTDVPSweepHandler
    const initialtime::Float64
    const finaltime::Float64
    const timestep::Float64
    ttn::TreeTensorNetwork{N, T}
    pTPO::ProjTPO_GPU
    func
    path::Vector{Tuple{Int,Int}}

    dirloop::Symbol # forward/backward-loop or topnode
    dir::Int #index of child for the next position in the path, 0 for parent node
    current_time::Float64
    imaginary_time::Bool
    # energy_shift::Float64
    # use_gpu::Bool false

    function TDVPSweepHandlerCPU(
        ttn::TreeTensorNetwork{N,T},
        pTPO,
        timestep,
        initialtime,
        finaltime,
        func,
        imaginary_time
    ) where {N,T}
        path = _tdvp_path(network(ttn))
        dir =
            path[2] ∈ child_nodes(network(ttn), path[1]) ?
            index_of_child(network(ttn), path[2]) : 0
        return new{N,T}(initialtime, finaltime, timestep, ttn, pTPO, func, path, :forward, dir, initialtime, imaginary_time)
    end
end

# forward mode of the TDVP sweep
# the tensor at position pos is only updated if the next step goes a layer up in the network, otherwise we just move the isometry center
function _tdvpforward!(sp::TDVPSweepHandlerCPU, pos::Tuple{Int,Int})
    ttn = sp.ttn
    pTPO = sp.pTPO
    net = network(ttn)
    
    # detmermine next position
    nextpos = sp.dir > 0 ? child_nodes(net, pos)[sp.dir] : parent_node(net, pos)
    Δ = nextpos .- pos

    T = ttn[pos]
    Tnext = ttn[nextpos]

    # if going down, just move ortho center to the next tensor and update environment
    if Δ[1] == -1
        # orthogonalize to child
        n_child = index_of_child(net, nextpos)
        _orthogonalize_to_child!(ttn, pos, n_child)
        #update environments
        
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = false)
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    # if going up perform time step algorithm
    elseif Δ[1] == 1
        # effective Hamiltonian for tensor at pos
        action = ∂A_GPU(pTPO, pos; use_gpu = false)
        (Tn,_) = sp.func(action, sp.timestep/2, T) 

        # QR-decompose time evolved tensor at pos
        idx_r = commonind(T, Tnext)
        idx_l = uniqueinds(T, idx_r)
        Qn,R = factorize(Tn, idx_l; tags = tags(idx_r))
        
        # reverse time evolution for link tensor between pos and next_pos
        action2 = ∂A2_GPU(pTPO, Qn, pos; use_gpu = false)
        (Rn,_) = sp.func(action2, -sp.timestep/2, R)

        # multiply new R tensor onto tensor at nextpos
        nextTn = Rn * Tnext
        # renormalize if imaginary time evolution
        sp.imaginary_time && (nextTn = nextTn/LinearAlgebra.norm(nextTn))

        ttn[pos] = Qn
        ttn[nextpos] = nextTn

        # move orthocenter (just for consistency), update ttnc and environments
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = false)
    else
        error("Invalid direction for TDVP step: ", Δ)
    end
end

# backward mode of the TDVP sweep
# the tensor at position pos is only updated if the next step goes a layer down in the network, otherwise we just move the isometry center
function _tdvpbackward!(sp::TDVPSweepHandlerCPU, pos::Tuple{Int,Int})
    ttn = sp.ttn
    pTPO = sp.pTPO
    net = network(ttn)

    # detmermine next position
    nextpos = sp.dir > 0 ? child_nodes(net, pos)[sp.dir] : parent_node(net, pos)
    Δ = nextpos .- pos

    T = ttn[pos]
    Tnext = ttn[nextpos]

    # if going up, just move ortho center to the next tensor and update environment
    if Δ[1] == 1
        # orthogonalize to parent
        
        _orthogonalize_to_parent!(ttn, pos)

        #update environments and ortho center
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = false)
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    # if going down perform time step algorithm
    elseif Δ[1] == -1
        # QR decomposition in direction of the TDVP-step
        idx_r = commonind(T, Tnext)
        idx_l = uniqueinds(T, idx_r)
        Qn, L = factorize(T, idx_l; tags = tags(idx_r))
        
        ttn[pos] = Qn

        # reverse time evolution for R tensor between pos and nextpos

        action = ∂A2_GPU(pTPO, Qn, pos; use_gpu = false)
        (Ln,_) = sp.func(action, -sp.timestep/2, L) 

        # multiply new L tensor on tensor at nextpos
        nextQ = Tnext* Ln
        # update environments & time evolve tensor at nextpos
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = false)
        action2 = ∂A_GPU(pTPO, nextpos; use_gpu = false)
        # println("Next comes func")
        (nextTn, _) = sp.func(action2, sp.timestep / 2, nextQ)

        # renormalize if imaginary time evolution
        sp.imaginary_time && (nextTn = nextTn/LinearAlgebra.norm(nextTn))

        # set new tensor and move orthocenter (just for consistency)
        ttn[nextpos] = nextTn

        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    else
        error("Invalid direction for TDVP step: ", Δ[1])
    end
end

# time evolution of top-node
function _tdvptopnode!(sp::TDVPSweepHandlerCPU, pos::Tuple{Int,Int})
    ttn = sp.ttn
    pTPO = sp.pTPO

    T = ttn[pos]

    action = ∂A_GPU(pTPO, pos; use_gpu = false)
    (Tn, _) = sp.func(action, sp.timestep, T)

    # renormalize if imaginary time evolution
    sp.imaginary_time && (Tn = Tn/LinearAlgebra.norm(Tn))

    ttn[pos] = Tn
end

# kwargs for being compatible with additional arguments
function update!(sp::TDVPSweepHandlerCPU, pos::Tuple{Int,Int}; kwargs...)

    if sp.dirloop == :forward
        _tdvpforward!(sp, pos)
    elseif sp.dirloop == :topnode
        _tdvptopnode!(sp, pos)
    elseif sp.dirloop == :backward
        _tdvpbackward!(sp, pos)
    end
end