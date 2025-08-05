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

    T = node_cache[pos]
    # Tnext = haskey(node_cache, nextpos) ? node_cache[nextpos] : gpu(ttn[nextpos])
    if haskey(node_cache, nextpos)
        Tnext = node_cache[nextpos]
    else
        Tnext = gpu(ttn[nextpos])
        # Attach safe finalizer to see when tensor is collected
        finalizer(Tnext) do x
            @async println("Finalizer: node_cache[$pos] was collected.")
        end
        println("Start from pos $pos: Load tensor at nextpos: $nextpos in forward")
    end

    # if going down, just move ortho center to the next tensor and update environment
    if Δ[1] == -1
        # orthogonalize to child
        n_child = index_of_child(net, nextpos)
        _orthogonalize_to_child!(ttn, pos, n_child, node_cache)
        #update environments
        
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = true)
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    # if going up perform time step algorithm
    elseif Δ[1] == 1
        # effective Hamiltonian for tensor at pos
        action = ∂A_GPU(pTPO, pos; use_gpu = true)
        (Tn,_) = sp.func(action, sp.timestep/2, T) 

        # QR-decompose time evolved tensor at pos
        idx_r = commonind(T, Tnext)
        idx_l = uniqueinds(T, idx_r)
        Qn,R = factorize(Tn, idx_l; tags = tags(idx_r))
        
        # reverse time evolution for link tensor between pos and next_pos
        action2 = ∂A2_GPU(pTPO, Qn, pos; use_gpu = true)
        (Rn,_) = sp.func(action2, -sp.timestep/2, R)

        # multiply new R tensor onto tensor at nextpos
        nextTn = Rn * Tnext  
        ttn[pos] = cpu(Qn)
        node_cache[pos] = Qn

        ttn[nextpos] = cpu(nextTn)
        node_cache[nextpos] = nextTn

        # move orthocenter (just for consistency), update ttnc and environments
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = true)
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

    T = node_cache[pos]
    # Tnext = haskey(node_cache, nextpos) ? node_cache[nextpos] : gpu(ttn[nextpos])
    if haskey(node_cache, nextpos)
        Tnext = node_cache[nextpos]
    else
        Tnext = gpu(ttn[nextpos])
        # Attach safe finalizer to see when tensor is collected
        finalizer(Tnext) do x
            @async println("Finalizer: node_cache[$pos] was collected.")
        end
        println("Start from pos $pos: Load at nextpos: $nextpos in backward")
    end

    # if going up, just move ortho center to the next tensor and update environment
    if Δ[1] == 1
        # orthogonalize to parent
        
        _orthogonalize_to_parent!(ttn, pos, node_cache)

        #update environments and ortho center
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = true)
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    # if going down perform time step algorithm
    elseif Δ[1] == -1
        # QR decomposition in direction of the TDVP-step
        idx_r = commonind(T, Tnext)
        idx_l = uniqueinds(T, idx_r)
        Qn, L = factorize(T, idx_l; tags = tags(idx_r))
        
        ttn[pos] = cpu(Qn)
        node_cache[pos] = Qn
        
        # reverse time evolution for R tensor between pos and nextpos

        action = ∂A2_GPU(pTPO, Qn, pos; use_gpu = true)
        (Ln,_) = sp.func(action, -sp.timestep/2, L) 

        # multiply new L tensor on tensor at nextpos
        nextQ = Tnext* Ln
        # update environments & time evolve tensor at nextpos
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = true)
        action2 = ∂A_GPU(pTPO, nextpos; use_gpu = true)
        # println("Next comes func")
        (nextTn, _) = sp.func(action2, sp.timestep / 2, nextQ)

        # set new tensor and move orthocenter (just for consistency)
        ttn[nextpos] = cpu(nextTn)
        node_cache[nextpos] = nextTn

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

    T = node_cache[pos]

    action = ∂A_GPU(pTPO, pos; use_gpu = true)
    (Tn, _) = sp.func(action, sp.timestep, T)
    
    ttn[pos] = cpu(Tn)
    node_cache[pos] = Tn
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
