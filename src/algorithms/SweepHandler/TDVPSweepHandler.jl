mutable struct TDVPSweepHandler{N<:AbstractNetwork,T} <: AbstractRegularSweepHandler
    const initialtime::Float64
    const finaltime::Float64
    const timestep::Float64
    ttn::TreeTensorNetwork{N, T}
    pTPO::AbstractProjTPO
    func
    path::Vector{Tuple{Int,Int}}

    dirloop::Symbol # forward/backward-loop or topnode
    dir::Int #index of child for the next position in the path, 0 for parent node
    current_time::Float64

    function TDVPSweepHandler(
        ttn::TreeTensorNetwork{N,T},
        pTPO,
        timestep,
        initialtime,
        finaltime,
        func,
    ) where {N,T}
        path = _tdvp_path(network(ttn))
        dir =
            path[2] ∈ child_nodes(network(ttn), path[1]) ?
            index_of_child(network(ttn), path[2]) : 0
        return new{N,T}(initialtime, finaltime, timestep, ttn, pTPO, func, path, :forward, dir, initialtime)
    end
end

current_sweep(sh::TDVPSweepHandler) = sh.current_time

# iterating through the ttn
function Base.iterate(sp::TDVPSweepHandler)
    pos = start_position(sp)
    return (pos, 1)
end

function Base.iterate(sp::TDVPSweepHandler, state)
    (next_pos, next_state) = next_position(sp, state)
    if isnothing(next_pos)
        update_next_sweep!(sp)
        return nothing
    end
    return (next_pos, next_state)
end

# return time sweeps
sweeps(sp::TDVPSweepHandler) = (sp.initialtime):(sp.timestep):(sp.finaltime)
# initial position of the sweep
start_position(sp::TDVPSweepHandler) = (sp.path[1])
initialize!(::TDVPSweepHandler) = nothing

# update the current time of the sweep
function update_next_sweep!(sp::TDVPSweepHandler)
    sp.current_time += sp.timestep 
    return sp
end

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

# forward mode of the TDVP sweep
# the tensor at position pos is only updated if the next step goes a layer up in the network, otherwise we just move the isometry center
function _tdvpforward!(sp::TDVPSweepHandler, pos::Tuple{Int,Int})
    ttn = sp.ttn
    pTPO = sp.pTPO
    net = network(ttn)
    T = ttn[pos]
    println(" ========================== forward ======================= ")
    @show pos
    # detmermine next position
    nextpos = sp.dir > 0 ? child_nodes(net, pos)[sp.dir] : parent_node(net, pos)
    Δ = nextpos .- pos

    # if going down, just move ortho center to the next tensor and update environment
    if Δ[1] == -1
        # orthogonalize to child
        n_child = index_of_child(net, nextpos)
        _orthogonalize_to_child!(ttn, pos, n_child)

        #update environments
        pTPO = update_environments!(pTPO, ttn[pos], pos, nextpos)
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    # if going up perform time step algorithm
    elseif Δ[1] == 1
        # effective Hamiltonian for tensor at pos
        action = ∂A(pTPO, pos)
        (Tn,_) = sp.func(action, sp.timestep/2, T) 

        # QR-decompose time evolved tensor at pos
        idx_r = commonind(ttn[pos], ttn[nextpos])
        idx_l = uniqueinds(ttn[pos], idx_r)
        Qn,R = factorize(Tn, idx_l; tags = tags(idx_r))
        
        # reverse time evolution for link tensor between pos and nextpos
        action2 = ∂A2(pTPO, Qn, pos)
        (Rn,_) = sp.func(action2, -sp.timestep/2, R)

        # multiply new R tensor onto tensor at nextpos
        nextTn = Rn * ttn[nextpos]

        # set new tensors
        ttn[pos] = Qn
        ttn[nextpos] = nextTn

        # move orthocenter (just for consistency), update ttnc and environments
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]
        pTPO = update_environments!(pTPO, ttn[pos], pos, nextpos)
    else
        error("Invalid direction for TDVP step: ", Δ)
    end
end

# backward mode of the TDVP sweep
# the tensor at position pos is only updated if the next step goes a layer down in the network, otherwise we just move the isometry center
function _tdvpbackward!(sp::TDVPSweepHandler, pos::Tuple{Int,Int})
    println(" ========================== backward ======================= ")
    @show pos
    ttn = sp.ttn
    pTPO = sp.pTPO
    net = network(ttn)
    T = ttn[pos]
    # detmermine next position
    nextpos = sp.dir > 0 ? child_nodes(net, pos)[sp.dir] : parent_node(net, pos)
    Δ = nextpos .- pos

    # if going up, just move ortho center to the next tensor and update environment
    if Δ[1] == 1
        # orthogonalize to parent
        _orthogonalize_to_parent!(ttn, pos)

        #update environments and ortho center
        pTPO = update_environments!(pTPO, ttn[pos], pos, nextpos)
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    # if going down perform time step algorithm
    elseif Δ[1] == -1
        # QR decomposition in direction of the TDVP-step
        idx_r = commonind(T, ttn[nextpos])
        idx_l = uniqueinds(T, idx_r)
        Qn, L = factorize(T, idx_l; tags = tags(idx_r))
        ttn[pos] = Qn

        # reverse time evolution for R tensor between pos and nextpos
        action = ∂A2(pTPO, Qn, pos)
        (Ln,_) = sp.func(action, -sp.timestep/2, L) 

        # multiply new L tensor on tensor at nextpos
        nextQ = ttn[nextpos] * Ln

        # update environments & time evolve tensor at nextpos
        pTPO = update_environments!(pTPO, Qn, pos, nextpos)
        action2 = ∂A(pTPO, nextpos)
        (nextTn, _) = sp.func(action2, sp.timestep / 2, nextQ)

        # set new tensor and move orthocenter (just for consistency)
        ttn[nextpos] = nextTn
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    else
        error("Invalid direction for TDVP step: ", Δ[1])
    end
end

# time evolution of top-node
function _tdvptopnode!(sp::TDVPSweepHandler, pos::Tuple{Int,Int})
    ttn = sp.ttn
    pTPO = sp.pTPO
    T = ttn[pos]

    action = ∂A(pTPO, pos)
    (Tn, _) = sp.func(action, sp.timestep, T)
    ttn[pos] = Tn
end

# kwargs for being compatible with additional arguments
function update!(sp::TDVPSweepHandler, pos::Tuple{Int,Int}; kwargs...)
    if sp.dirloop == :forward
        _tdvpforward!(sp, pos)
    elseif sp.dirloop == :topnode
        _tdvptopnode!(sp, pos)
    elseif sp.dirloop == :backward
        _tdvpbackward!(sp, pos)
    end
end

# returns the next position in the sweep, based on the current mode and position
function next_position(sp::TDVPSweepHandler, state::Int)
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

#=
function _tdvpforward!(sp::TDVPSweepHandler{N,TensorMap}, pos::Tuple{Int,Int}) where {N}

    ttn = sp.ttn
    pTPO = sp.pTPO
    net = network(ttn)
    T = ttn[pos]
    nextpos = sp.dir > 0 ? child_nodes(net, pos)[sp.dir] : parent_node(net, pos)

    Δ = nextpos .- pos

    if Δ[1] == -1
        # orthogonalize to child
        n_child = index_of_child(net, nextpos)
        _orthogonalize_to_child!(ttn, pos, n_child)

        #update environments
        pTPO = update_environments!(pTPO, ttn[pos], pos, nextpos)

        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    elseif Δ[1] == 1

        # time evolve tensor at pos
        action = ∂A(pTPO, pos)
        (Tn, _) = sp.func(action, sp.timestep / 2, T)

        # QR-decompose time evolved tensor at pos
        Qn, R = leftorth(Tn)

        # reverse time evolution for R tensor between pos and nextpos
        action2 = ∂A2(pTPO, Qn, pos, nextpos)
        (Rn, _) = sp.func(action2, -sp.timestep / 2, R)

        # multiply new R tensor onto tensor at nextpos
        idx = index_of_child(net, pos)
        idx_dom, idx_codom = split_index(net, nextpos, idx)
        perm = vcat(idx_dom..., idx_codom...)
        nextT = TensorKit.permute(ttn[nextpos], idx_dom, idx_codom)
        nextTn = TensorKit.permute(Rn * nextT, Tuple(perm[1:(end - 1)]), (perm[end],))

        # set new tensors
        ttn[pos] = Qn
        ttn[nextpos] = nextTn

        # move orthocenter (just for consistency), update ttnc and environments
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

        pTPO = update_environments!(pTPO, ttn[pos], pos, nextpos)

    else
        error("Invalid direction for TDVP step: ", Δ)
    end
end

function _tdvpbackward!(sp::TDVPSweepHandler{N,TensorMap}, pos::Tuple{Int,Int}) where {N}

    ttn = sp.ttn
    pTPO = sp.pTPO
    net = network(ttn)
    T = ttn[pos]
    nextpos = sp.dir > 0 ? child_nodes(net, pos)[sp.dir] : parent_node(net, pos)

    Δ = nextpos .- pos

    if Δ[1] == 1
        # orthogonalize to parent
        _orthogonalize_to_parent!(ttn, pos)

        #update environments
        pTPO = update_environments!(pTPO, ttn[pos], pos, nextpos)

        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    elseif Δ[1] == -1

        # QR decomposition in direction of the TDVP-step
        idx_chd = index_of_child(net, nextpos)
        idx_dom, idx_codom = TTNKit.split_index(net, pos, idx_chd)
        perm = vcat(idx_dom..., idx_codom...)
        L, Qn = rightorth(T, idx_dom, idx_codom)
        Qn = TensorKit.permute(Qn, Tuple(perm[1:(end - 1)]), (perm[end],))
        ttn[pos] = Qn

        # reverse time evolution for R tensor between pos and nextpos
        action = ∂A2(pTPO, Qn, pos, nextpos)
        (Ln, _) = sp.func(action, -sp.timestep / 2, L)

        # multiply new L tensor on tensor at nextpos
        nextQ = ttn[nextpos] * Ln

        # update environments & time evolve tensor at nextpos
        pTPO = update_environments!(pTPO, Qn, pos, nextpos)
        action2 = ∂A(pTPO, nextpos)
        (nextTn, _) = sp.func(action2, sp.timestep / 2, nextQ)

        # set new tensor and move orthocenter (just for consistency)
        ttn[nextpos] = nextTn
        ttn.ortho_center[1] = nextpos[1]
        ttn.ortho_center[2] = nextpos[2]

    else
        error("Invalid direction for TDVP step: ", Δ[1])
    end
end
=#
