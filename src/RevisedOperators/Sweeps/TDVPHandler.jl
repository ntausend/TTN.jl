mutable struct TDVPSweepHandlerGPU{N<:AbstractNetwork,T} <: AbstractRegularSweepHandler
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
    use_gpu::Bool

    function TDVPSweepHandlerGPU(
        ttn::TreeTensorNetwork{N,T},
        pTPO,
        timestep,
        initialtime,
        finaltime,
        func,
        use_gpu
    ) where {N,T}
        path = _tdvp_path(network(ttn))
        dir =
            path[2] ∈ child_nodes(network(ttn), path[1]) ?
            index_of_child(network(ttn), path[2]) : 0
        return new{N,T}(initialtime, finaltime, timestep, ttn, pTPO, func, path, :forward, dir, initialtime, use_gpu)
    end
end

current_sweep(sh::TDVPSweepHandlerGPU) = sh.current_time

# iterating through the ttn
function Base.iterate(sp::TDVPSweepHandlerGPU)
    pos = start_position(sp)
    return (pos, 1)
end

function Base.iterate(sp::TDVPSweepHandlerGPU, state)
    (next_pos, next_state) = next_position(sp, state)
    if isnothing(next_pos)
        update_next_sweep!(sp)
        return nothing
    end
    return (next_pos, next_state)
end

# return time sweeps
sweeps(sp::TDVPSweepHandlerGPU) = (sp.initialtime):(sp.timestep):(sp.finaltime)
# initial position of the sweep
start_position(sp::TDVPSweepHandlerGPU) = (sp.path[1])
initialize!(::TDVPSweepHandlerGPU) = nothing

# update the current time of the sweep
function update_next_sweep!(sp::TDVPSweepHandlerGPU)
    sp.current_time += sp.timestep 
    return sp
end

#=
# path of a single TDVP update through the TTN
# the path is split into a forward and a backward mode, as well as a single separate update for the top node
function _tdvp_path(net::AbstractNetwork)
    path = Vector{Tuple{Int,Int}}([(number_of_layers(net), 1)])
    function gotochild(pos::Tuple{Int,Int})      
        for chd_nd in child_nodes(net, pos)
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
=#

# forward mode of the TDVP sweep
# the tensor at position pos is only updated if the next step goes a layer up in the network, otherwise we just move the isometry center
function _tdvpforward!(sp::TDVPSweepHandlerGPU, pos::Tuple{Int,Int}; node_cache = Dict())
    ttn = sp.ttn
    pTPO = sp.pTPO
    net = network(ttn)
    use_gpu = sp.use_gpu
    
    # detmermine next position
    nextpos = sp.dir > 0 ? child_nodes(net, pos)[sp.dir] : parent_node(net, pos)
    Δ = nextpos .- pos

    if use_gpu
        T = haskey(node_cache, pos) ? node_cache[pos] : gpu(ttn[pos])
        Tnext = haskey(node_cache, nextpos) ? node_cache[nextpos] : gpu(ttn[nextpos])
    else
        T = ttn[pos]
        Tnext = ttn[nextpos]
    end

    # if going down, just move ortho center to the next tensor and update environment
    if Δ[1] == -1
        # orthogonalize to child
        n_child = index_of_child(net, nextpos)
        if use_gpu
            _orthogonalize_to_child!(ttn, pos, n_child, node_cache)
        else 
            _orthogonalize_to_child!(ttn, pos, n_child)
        end
        #update environments
        
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = use_gpu)
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    # if going up perform time step algorithm
    elseif Δ[1] == 1
        # effective Hamiltonian for tensor at pos
        action = ∂A_GPU(pTPO, pos; use_gpu = use_gpu)
        (Tn,_) = sp.func(action, sp.timestep/2, T) 

        # QR-decompose time evolved tensor at pos
        idx_r = commonind(T, Tnext)
        idx_l = uniqueinds(T, idx_r)
        Qn,R = factorize(Tn, idx_l; tags = tags(idx_r))
        
        # reverse time evolution for link tensor between pos and next_pos
        # println(Tn)
        # println(Qn)
        # println(pos)
        action2 = ∂A2_GPU(pTPO, Qn, pos; use_gpu = use_gpu)
        (Rn,_) = sp.func(action2, -sp.timestep/2, R)

        # multiply new R tensor onto tensor at nextpos
        nextTn = Rn * Tnext
        if use_gpu        
            ttn[pos] = cpu(Qn)
            node_cache[pos] = Qn

            ttn[nextpos] = cpu(nextTn)
            node_cache[nextpos] = nextTn
        else
            ttn[pos] = Qn
            ttn[nextpos] = nextTn
        end

        # move orthocenter (just for consistency), update ttnc and environments
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = use_gpu)
    else
        error("Invalid direction for TDVP step: ", Δ)
    end
end

# backward mode of the TDVP sweep
# the tensor at position pos is only updated if the next step goes a layer down in the network, otherwise we just move the isometry center
function _tdvpbackward!(sp::TDVPSweepHandlerGPU, pos::Tuple{Int,Int}; node_cache = Dict())
    ttn = sp.ttn
    pTPO = sp.pTPO
    net = network(ttn)
    use_gpu = sp.use_gpu

    # detmermine next position
    nextpos = sp.dir > 0 ? child_nodes(net, pos)[sp.dir] : parent_node(net, pos)
    Δ = nextpos .- pos

    if use_gpu
        T = haskey(node_cache, pos) ? node_cache[pos] : gpu(ttn[pos])
        Tnext = haskey(node_cache, nextpos) ? node_cache[nextpos] : gpu(ttn[nextpos])
    else
        T = ttn[pos]
        Tnext = ttn[nextpos]
    end

    # if going up, just move ortho center to the next tensor and update environment
    if Δ[1] == 1
        # orthogonalize to parent
        
        if use_gpu
            _orthogonalize_to_parent!(ttn, pos, node_cache)
        else 
            _orthogonalize_to_parent!(ttn, pos)
        end

        #update environments and ortho center
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = use_gpu)
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    # if going down perform time step algorithm
    elseif Δ[1] == -1
        # QR decomposition in direction of the TDVP-step
        idx_r = commonind(T, Tnext)
        idx_l = uniqueinds(T, idx_r)
        Qn, L = factorize(T, idx_l; tags = tags(idx_r))
        if use_gpu
            ttn[pos] = cpu(Qn)
            node_cache[pos] = Qn
        else
            ttn[pos] = Qn
        end
        # println("pos inds", inds(T))
        # println("nextpos inds", inds(Tnext))
        # println("Qn inds", inds(Qn))
        # println("L inds", inds(L))

        # reverse time evolution for R tensor between pos and nextpos

        action = ∂A2_GPU(pTPO, Qn, pos; use_gpu = use_gpu)
        (Ln,_) = sp.func(action, -sp.timestep/2, L) 

        # multiply new L tensor on tensor at nextpos
        nextQ = Tnext* Ln
        # println("nextQ inds", inds(nextQ))
        # println("Position = $pos Next position = $nextpos")
        # update environments & time evolve tensor at nextpos
        pTPO = recalc_path_flows!(pTPO, ttn, pos, nextpos; use_gpu = use_gpu)
        action2 = ∂A_GPU(pTPO, nextpos; use_gpu = use_gpu)
        # println("Next comes func")
        (nextTn, _) = sp.func(action2, sp.timestep / 2, nextQ)

        # set new tensor and move orthocenter (just for consistency)
        if use_gpu
            ttn[nextpos] = cpu(nextTn)
            node_cache[nextpos] = nextTn
        else
            ttn[nextpos] = nextTn
        end

        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    else
        error("Invalid direction for TDVP step: ", Δ[1])
    end
end

# time evolution of top-node
function _tdvptopnode!(sp::TDVPSweepHandlerGPU, pos::Tuple{Int,Int}; node_cache = Dict())
    ttn = sp.ttn
    pTPO = sp.pTPO
    use_gpu = sp.use_gpu

    if use_gpu
        T = haskey(node_cache, pos) ? node_cache[pos] : gpu(ttn[pos])
    else
        T = ttn[pos]
    end

    action = ∂A_GPU(pTPO, pos; use_gpu = use_gpu)
    (Tn, _) = sp.func(action, sp.timestep, T)

    if use_gpu        
        ttn[pos] = cpu(Tn)
        node_cache[pos] = Tn
    else
        ttn[pos] = Tn
    end

end

# kwargs for being compatible with additional arguments
function update!(sp::TDVPSweepHandlerGPU, pos::Tuple{Int,Int}; node_cache= Doct(), kwargs...)
    if sp.dirloop == :forward
        # println("Forward", pos)
        _tdvpforward!(sp, pos; node_cache = node_cache)
    elseif sp.dirloop == :topnode
        # println("Topnode", pos)
        _tdvptopnode!(sp, pos; node_cache = node_cache)
    elseif sp.dirloop == :backward
        # println("Backward", pos)
        _tdvpbackward!(sp, pos; node_cache = node_cache)
    end
end

# returns the next position in the sweep, based on the current mode and position
function next_position(sp::TDVPSweepHandlerGPU, state::Int)
    len = length(sp.path) - 1
    net = network(sp.ttn)

    if sp.dirloop == :forward
        if state == len
            sp.dirloop = :topnode
            return (sp.path[state + 1], 1)
        end
        sp.dir =
            sp.path[state + 2] ∈ child_nodes(net, sp.path[state + 1]) ?
            index_of_child(net, sp.path[state + 2]) : 0
        return (sp.path[state + 1], state + 1)
    elseif sp.dirloop == :topnode
        sp.dirloop = :backward
        sp.dir =
            reverse(sp.path)[2] ∈ child_nodes(net, reverse(sp.path)[1]) ?
            index_of_child(net, reverse(sp.path)[2]) : 0
        return (reverse(sp.path)[1], 1)
    elseif sp.dirloop == :backward
        if state == len
            sp.dirloop = :forward
            sp.dir =
                sp.path[2] ∈ child_nodes(net, sp.path[1]) ?
                index_of_child(net, sp.path[2]) : 0
            return (nothing, 1)
        end
        sp.dir =
            reverse(sp.path)[state + 2] ∈ child_nodes(net, reverse(sp.path)[state + 1]) ?
            index_of_child(net, reverse(sp.path)[state + 2]) : 0
        return (reverse(sp.path)[state + 1], state + 1)
    end
    error("Invalid direction of the iterator: $(sp.dirloop)")
end

