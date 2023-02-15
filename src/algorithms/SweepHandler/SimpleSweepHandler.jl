mutable struct SimpleSweepHandler <: AbstractRegularSweepHandler
    const number_of_sweeps::Int
    ttn::TreeTensorNetwork
    pTPO::AbstractProjTPO
    func
    expander::AbstractSubspaceExpander
        
    maxdims::Vector{Int64}
    noise::Vector{Number}

    dir::Symbol
    current_sweep::Int
    current_energy::Float64
    current_spec::Spectrum
    current_max_truncerr::Float64
    SimpleSweepHandler(ttn, pTPO, func, n_sweeps, maxdims, noise, expander = NoExpander()) = 
        new(n_sweeps, ttn, pTPO, func, expander, maxdims, noise, :up, 1, 0., Spectrum(nothing, 0.0), 0.0)
end

current_sweep(sh::SimpleSweepHandler) = sh.current_sweep

function noise(sh::SimpleSweepHandler)
    length(sh.noise) < current_sweep(sh) && return sh.noise[end]
    return sh.noise[current_sweep(sh)]
end

function maxdim(sh::SimpleSweepHandler)
    length(sh.maxdims) < current_sweep(sh) && return sh.maxdims[end]
    return sh.maxdims[current_sweep(sh)]
end

function info_string(sh::SimpleSweepHandler, output_level::Int)
    e = sh.current_energy
    trnc_wght = sh.current_max_truncerr
    # todo ->  make a function for that .... which also can handle TensorKit
    maxdim = maxlinkdim(sh.ttn)
    output_level ≥ 1 && @printf("\tCurrent energy: %.6f.\n", e)
    output_level ≥ 2 && @printf("\tTruncated Weigth: %.3e. Maximal bond dim = %i\n", trnc_wght, maxdim)
    sh.current_max_truncerr = 0.0
    nothing
end

function initialize!(sp::SimpleSweepHandler)
    ttn = sp.ttn
    pTPO = sp.pTPO

    #adjust the tree dimension to the first bond dimension

    # move to starting point of the sweep
    ttn = move_ortho!(ttn, (1,1))
    # update the environments accordingly
    pTPO = set_position!(pTPO, ttn)

    sp.ttn = ttn
    sp.pTPO = pTPO
    # get starting energy
    return sp
end

# simple reset the sweep Handler and update the current sweep number
# current number still needed?
function update_next_sweep!(sp::SimpleSweepHandler)
    sp.dir = :up
    sp.current_sweep += 1 
    return sp
end

function update!(sp::SimpleSweepHandler, pos::Tuple{Int, Int})
    @assert pos == ortho_center(sp.ttn)
    ttn = sp.ttn
    pTPO = sp.pTPO
    
    net = network(ttn)

    # adjust the position of the projected operator, i.e. redefine the environemnts
    pTPO = set_position!(pTPO, ttn)

    t = ttn[pos]
    
    
    #=
    #if sp.dir == :up
        prnt_nd = parent_node(net, pos)
        # only do the subspace expansion if going upwards in the tree
        if isnothing(prnt_nd)
            # evolving of parent node, using the two childs
            chldnds = child_nodes(net, pos)
            length(chldnds) == 2 || error("Top node supspace expansion not implemented for non binary trees")
            A_chlds = map(p -> getindex(ttn, p), chldnds)
            Al, Ar = A_chlds
            ids_sh = commonind(Al, t)
            ids_l = uniqueinds(Al, ids_sh)

            tpr = Al*t
            tpr, Ar = expand(tpr, Ar, sp.expander; reorthogonalize = true)

            @show maximum(ITensors.dims(t))
            Al, t = factorize(tpr, ids_l; tags = tags(ids_sh))
            @show maximum(ITensors.dims(t))

            #t, Al, Ar = expand(t, Tuple(A_chlds), sp.expander; reorthogonalize = true)
            ttn[pos] = t
            ttn[chldnds[1]] = Al
            ttn[chldnds[2]] = Ar
            pTPO = update_environments!(pTPO, Al, chldnds[1], pos)
            pTPO = update_environments!(pTPO, Ar, chldnds[2], pos)
        else
            # do the expansion towards topnode
            A_prnt = ttn[prnt_nd]

            t, Aprime = expand(t, A_prnt, sp.expander; reorthogonalize = true)
            ttn[pos] = t
            ttn[prnt_nd] = Aprime
            pTPO = update_environments!(pTPO, Aprime, prnt_nd, pos)
        end
    #end
    pn = next_position(sp, pos)
    =#

      
    pn = next_position(sp,pos)
    pth = nothing
    if !isnothing(pn)
        pth = connecting_path(net, pos, pn) 
        posnext = pth[1]
        # do a subspace expansion, in case of being qn symmetric
        if length(inds(t)) > 2
            A_next = ttn[posnext]
            #
            t, Aprime = expand(t, A_next, sp.expander; reorthogonalize = true)

            ttn[pos]     = t
            ttn[posnext] = Aprime
            # this does not change the orthogonality center of the pTPO, since
            # we only updating the old isometry with the new one fitting to the
            # expanded state
            pTPO = update_environments!(pTPO, Aprime, posnext, pos)
        else
            # evolving of parent node, using the two childs
            chldnds = child_nodes(net, pos)
            length(chldnds) == 2 || error("Top node supspace expansion not implemented for non binary trees")
            A_chlds = map(p -> getindex(ttn, p), chldnds)
            Al, Ar = A_chlds
            ids_sh = commonind(Al, t)
            ids_l = uniqueinds(Al, ids_sh)

            tpr = Al*t
            tpr, Ar = expand(tpr, Ar, sp.expander; reorthogonalize = true)

            Al, t = factorize(tpr, ids_l; tags = tags(ids_sh))

            #t, Al, Ar = expand(t, Tuple(A_chlds), sp.expander; reorthogonalize = true)
            ttn[pos] = t
            ttn[chldnds[1]] = Al
            ttn[chldnds[2]] = Ar
            pTPO = update_environments!(pTPO, Al, chldnds[1], pos)
            pTPO = update_environments!(pTPO, Ar, chldnds[2], pos)
        end
    end
    


    action  = ∂A(pTPO, pos)
    val, tn = sp.func(action, t)
    sp.current_energy = real(val[1])
    tn = tn[1]

    # building the noise term
    
    drho = nothing
    if noise(sp) > 0 && !isnothing(pth)
        drho = noiseterm(pTPO, tn, pth[1])
    end

    # possible other arguments, which_decomp, svd_alg etc
    ttn, spec = update_node_and_move!(ttn,tn, pn;
                                      maxdim = maxdim(sp),
                                      eigen_perturbation = drho,
                                      normalize = true)

    sp.current_spec = spec
    trncerr = truncerror(spec)
    sp.current_max_truncerr = max(sp.current_max_truncerr, trncerr)
    return sp
end


function next_position(sp::SimpleSweepHandler, cur_pos::Tuple{Int,Int})
    cur_layer, cur_p = cur_pos
    net = network(sp.ttn)
    if sp.dir == :up
        max_pos = number_of_tensors(net, cur_layer)
        cur_p < max_pos && return (cur_layer, cur_p + 1)
        if cur_layer == number_of_layers(net)
            sp.dir = :down
            return (cur_layer - 1, number_of_tensors(net, cur_layer - 1))
        end
        return (cur_layer + 1, 1)
    elseif sp.dir == :down
        cur_p > 1 && return (cur_layer, cur_p - 1)
        cur_layer == 1 && return nothing
        return (cur_layer - 1, number_of_tensors(net, cur_layer - 1))
    end
    error("Invalid direction of the iterator: $(sp.dir)")
end
