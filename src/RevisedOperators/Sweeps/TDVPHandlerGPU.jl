mutable struct TDVPSweepHandlerGPU{N<:AbstractNetwork,T} <: AbstractTDVPSweepHandler
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
    # use_gpu::Bool true

    function TDVPSweepHandlerGPU(
        ttn::TreeTensorNetwork{N,T},
        pTPO,
        timestep,
        initialtime,
        finaltime,
        func) where {N,T}
        path = _tdvp_path(network(ttn))
        dir =
            path[2] ∈ child_nodes(network(ttn), path[1]) ?
            index_of_child(network(ttn), path[2]) : 0
        return new{N,T}(initialtime, finaltime, timestep, ttn, pTPO, func, path, :forward, dir, initialtime)
    end
end

# forward mode of the TDVP sweep
# the tensor at position pos is only updated if the next step goes a layer up in the network, otherwise we just move the isometry center
function _tdvpforward!(sp::TDVPSweepHandlerGPU, pos::Tuple{Int,Int}; node_cache::Dict)
    ttn = sp.ttn
    pTPO = sp.pTPO
    net = network(ttn)
    
    # detmermine next position
    nextpos = sp.dir > 0 ? child_nodes(net, pos)[sp.dir] : parent_node(net, pos)
    Δ = nextpos .- pos

    # T assumed to be already loaded in node_cache
    T = ttn[pos]
    # T_gpu = node_cache[pos]

    Tnext = ttn[nextpos]
    # Tnext = haskey(node_cache, nextpos) ? node_cache[nextpos] : gpu(ttn[nextpos])

    # if going down, just move ortho center to the next tensor and update environment
    if Δ[1] == -1
        # orthogonalize to child
        n_child = index_of_child(net, nextpos)
        _orthogonalize_to_child!(ttn, pos, n_child, node_cache)
        #update environments
        
        recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = true, node_cache = node_cache)
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    # if going up perform time step algorithm
    elseif Δ[1] == 1
        # effective Hamiltonian for tensor at pos
        # loaded on gpu if use_gpu == true
        action = ∂A_GPU(pTPO, pos; use_gpu = true)
        (Tn_gpu,_) = sp.func(action, sp.timestep/2, node_cache[pos]) # , T_gpu

        # delete T_gpu to free memory for next action
        delete!(node_cache, pos)

        # QR-decompose time evolved tensor at pos
        idx_r = commonind(T, Tnext)
        idx_l = uniqueinds(T, idx_r)
        Qn_gpu, R_gpu = factorize(Tn_gpu, idx_l; tags = tags(idx_r))
        
        # reverse time evolution for link tensor between pos and next_pos
        action2 = ∂A2_GPU(pTPO, Qn_gpu, pos; use_gpu = true)
        (Rn_gpu,_) = sp.func(action2, -sp.timestep/2, R_gpu)

        # multiply new R tensor onto tensor at nextpos
        ttn[pos] = cpu(Qn_gpu)
        node_cache[pos] = Qn_gpu

        nextTn_gpu = Rn_gpu * gpu(Tnext)
        ttn[nextpos] = cpu(nextTn_gpu)
        node_cache[nextpos] = nextTn_gpu
        # delete GPU tensors to free memory

        # move orthocenter (just for consistency), update ttnc and environments
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]
        recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = true, node_cache = node_cache)
    else
        error("Invalid direction for TDVP step: ", Δ)
    end
end

# backward mode of the TDVP sweep
# the tensor at position pos is only updated if the next step goes a layer down in the network, otherwise we just move the isometry center
function _tdvpbackward!(sp::TDVPSweepHandlerGPU, pos::Tuple{Int,Int}; node_cache::Dict)
    ttn = sp.ttn
    pTPO = sp.pTPO
    net = network(ttn)

    # detmermine next position
    nextpos = sp.dir > 0 ? child_nodes(net, pos)[sp.dir] : parent_node(net, pos)
    Δ = nextpos .- pos

    # T assumed to be already loaded in node_cache
    T = ttn[pos]
    # T_gpu = node_cache[pos]

    Tnext = ttn[nextpos]
    # Tnext = haskey(node_cache, nextpos) ? node_cache[nextpos] : gpu(ttn[nextpos])

    # if going up, just move ortho center to the next tensor and update environment
    if Δ[1] == 1
        # orthogonalize to parent
        
        _orthogonalize_to_parent!(ttn, pos, node_cache)

        #update environments and ortho center
        recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = true, node_cache = node_cache)
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    # if going down perform time step algorithm
    elseif Δ[1] == -1
        # QR decomposition in direction of the TDVP-step
        idx_r = commonind(T, Tnext)
        idx_l = uniqueinds(T, idx_r)
        Qn_gpu, L_gpu = factorize(node_cache[pos], idx_l; tags = tags(idx_r))
        
        ttn[pos] = cpu(Qn_gpu)

        # delete T_gpu to free memory for next action
        delete!(node_cache, pos)
        
        # reverse time evolution for R tensor between pos and nextpos

        action = ∂A2_GPU(pTPO, Qn_gpu, pos; use_gpu = true)
        (Ln_gpu,_) = sp.func(action, -sp.timestep/2, L_gpu)

        # multiply new L tensor on tensor at nextpos
        nextQ_gpu = gpu(Tnext)* Ln_gpu

        # update environments & time evolve tensor at nextpos
        node_cache[pos] = Qn_gpu
        recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = true, node_cache = node_cache)
        delete!(node_cache, pos)
        action2 = ∂A_GPU(pTPO, nextpos; use_gpu = true)
        (nextTn_gpu, _) = sp.func(action2, sp.timestep / 2, nextQ_gpu)
        # set new tensor and move orthocenter (just for consistency)
        ttn[nextpos] = cpu(nextTn_gpu)
        node_cache[nextpos] = nextTn_gpu

        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    else
        error("Invalid direction for TDVP step: ", Δ[1])
    end
end

# time evolution of top-node
function _tdvptopnode!(sp::TDVPSweepHandlerGPU, pos::Tuple{Int,Int}; node_cache::Dict)
    ttn = sp.ttn
    pTPO = sp.pTPO

    # T assumed to be already loaded in node_cache
    # T_gpu = node_cache[pos]

    action = ∂A_GPU(pTPO, pos; use_gpu = true)
    (Tn_gpu, _) = sp.func(action, sp.timestep, node_cache[pos])
    
    ttn[pos] = cpu(Tn_gpu)
    # node_cache[pos] = Tn_gpu
end

# kwargs for being compatible with additional arguments
function update!(sp::TDVPSweepHandlerGPU, pos::Tuple{Int,Int}; node_cache::Dict, kwargs...)

    if sp.dirloop == :forward
        _tdvpforward!(sp, pos; node_cache = node_cache)
    elseif sp.dirloop == :topnode
        _tdvptopnode!(sp, pos; node_cache = node_cache)
    elseif sp.dirloop == :backward
        _tdvpbackward!(sp, pos; node_cache = node_cache)
    end
end
