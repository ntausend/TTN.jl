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
    outputlevel::Int
    SimpleSweepHandler(ttn, pTPO, func, n_sweeps, maxdims, noise, expander = NoExpander(), outputlevel = 0) = 
        new(n_sweeps, ttn, pTPO, func, expander, maxdims, noise, :up, 1, 0., Spectrum(nothing, 0.0), 0.0, outputlevel)
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
    output_level ≥ 1 && @printf("\tCurrent energy: %.15f.\n", e)
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

function update!(sp::SimpleSweepHandler, pos::Tuple{Int, Int}; svd_alg = nothing)
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
    
    ttn[pos] = tn
    

    # if we did expansion, perform iteration over the expanded leg till convergence
    if sp.expander isa NonTrivialExpander && !isnothing(pth)
        
        # getting next tensor in the direction of the path -> enlarged tensor
        #posnext = first(connecting_path(net, pos, pn))
        posnext = first(pth)
        Tnext = ttn[posnext] 
        # move orthogonality center to Tnext
        id_sh = commonind(tn, Tnext)
        Q,R, spec, id_sh = factorize(tn, uniqueinds(tn, id_sh); tags = tags(id_sh))
        Tnext = R*Tnext
        tn = Q
        # update environments for minimizing w.r.t. Tnext
        pTPO = update_environments!(pTPO, Q, pos, posnext)
        
        # find the optimal Tnext tensor
        action = ∂A(pTPO, posnext)
        val, Tnext′ = sp.func(action, Tnext)
        sp.current_energy = real(first(val))
        Tnext′ = first(Tnext′)
        # difference to the prev iteration (norm of the two site tensor -> removes gauge degrees of freedom
        T2T = tn * Tnext′
        ϵ = norm(T2T - tn * Tnext)
        
        Tnext = Tnext′
        # move orthogonality center to tn again for new optimization
        #id_sh = commonind(Tnext, tn)
        Q,R, spec, id_sh = factorize(Tnext, uniqueinds(Tnext, id_sh); tags = tags(id_sh))
        Tnext = Q
        tn = tn * R
        pTPO = update_environments!(pTPO, Q, posnext, pos)

        ttn[pos] = tn
        ttn[posnext] = Tnext

        # now perform iterations till reaching convergence or max iteration number is reached
        curiter = 1
        while (curiter < maxiter(sp.expander)) && (ϵ > tol(sp.expander))
            
            action = ∂A(pTPO, pos)
            val, tn = sp.func(action, tn)
            sp.current_energy = real(first(val))
            tn = first(tn)
            
            Q,R, spec, id_sh = factorize(tn, uniqueinds(tn, id_sh); tags = tags(id_sh))
            Tnext = R * Tnext
            tn = Q
            # update environments for minimizing w.r.t. Tnext
            pTPO = update_environments!(pTPO, Q, pos, posnext)
            
            # find the optimal Tnext tensor
            action = ∂A(pTPO, posnext)
            val, Tnext = sp.func(action, Tnext)
            sp.current_energy = real(first(val))
            Tnext = first(Tnext)
            
            # difference to the prev iteration (norm of the two site tensor -> removes gauge degrees of freedom
            T2Tn = tn * Tnext
            ϵ = norm(T2T - T2Tn)
            T2T = T2Tn
            
            # move orthogonality center to tn again for new optimization
            #id_sh = commonind(Tnext, tn)
            Q,R, spec, id_sh = factorize(Tnext, uniqueinds(Tnext, id_sh); tags = tags(id_sh))
            Tnext = Q
            tn = tn * R
            pTPO = update_environments!(pTPO, Q, posnext, pos)

            curiter += 1
            if sp.outputlevel > 3
                println("\t\t\t Current iter $(curiter) of $(maxiter(sp.expander)). Current error: $(ϵ)")
            end

            ttn[pos] = tn
            ttn[posnext] = Tnext
        end
        if sp.outputlevel > 2
            println("\t\tStopped expansion loop after $(curiter) iterations of $(maxiter(sp.expander)). Final error: $(ϵ)")
        end

    end

    # building the noise term
    
    drho = nothing
    if noise(sp) > 0 && !isnothing(pth)
        drho = noiseterm(pTPO, tn, pth[1])
        #@info inds(drho)
        #@info inds(tn)
    end

    # possible other arguments, which_decomp, svd_alg etc
    ttn, spec = update_node_and_move!(ttn,tn, pn;
                                      maxdim = maxdim(sp),
                                      eigen_perturbation = drho,
                                      normalize = true,
                                      svd_alg)

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



###### This is the excited sweep handler stuff ########
mutable struct ExcitedSweepHandler <: AbstractRegularSweepHandler
    const number_of_sweeps::Int
    ttn::TreeTensorNetwork
    ttns_ortho::Vector{TreeTensorNetwork}
    pTPO::AbstractProjTPO
    func
    expander::AbstractSubspaceExpander
    weight::Float64
        
    maxdims::Vector{Int64}
    noise::Vector{Number}

    dir::Symbol
    current_sweep::Int
    current_energy::Float64
    current_spec::Spectrum
    current_max_truncerr::Float64
    ExcitedSweepHandler(ttn, ttns_ortho, pTPO, func, n_sweeps, maxdims, noise, expander = NoExpander(), weight=10.0) = 
        new(n_sweeps, ttn, ttns_ortho, pTPO, func, expander, weight, maxdims, noise, :up, 1, 0., Spectrum(nothing, 0.0), 0.0)
end


current_sweep(sh::ExcitedSweepHandler) = sh.current_sweep

function noise(sh::ExcitedSweepHandler)
    length(sh.noise) < current_sweep(sh) && return sh.noise[end]
    return sh.noise[current_sweep(sh)]
end

function maxdim(sh::ExcitedSweepHandler)
    length(sh.maxdims) < current_sweep(sh) && return sh.maxdims[end]
    return sh.maxdims[current_sweep(sh)]
end

function info_string(sh::ExcitedSweepHandler, output_level::Int)
    e = sh.current_energy
    trnc_wght = sh.current_max_truncerr
    # todo ->  make a function for that .... which also can handle TensorKit
    maxdim = maxlinkdim(sh.ttn)
    output_level ≥ 1 && @printf("\tCurrent energy: %.6f.\n", e)
    output_level ≥ 2 && @printf("\tTruncated Weigth: %.3e. Maximal bond dim = %i\n", trnc_wght, maxdim)
    sh.current_max_truncerr = 0.0
    nothing
end

function initialize!(sp::ExcitedSweepHandler)
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
function update_next_sweep!(sp::ExcitedSweepHandler)
    sp.dir = :up
    sp.current_sweep += 1 
    return sp
end

function update!(sp::ExcitedSweepHandler, pos::Tuple{Int, Int})
    @assert pos == ortho_center(sp.ttn)
    ttn = sp.ttn
    pTPO = sp.pTPO
    
    net = network(ttn)

    # adjust the position of the projected operator, i.e. redefine the environemnts
    pTPO = set_position!(pTPO, ttn)

    # adjust ortho_centers of the orthogonal states to match the current ortho center
    for (i,ttn_ortho) in enumerate(sp.ttns_ortho)
        sp.ttns_ortho[i] = move_ortho!(ttn_ortho, pos)
        @assert ortho_center(ttn_ortho) == pos
    end

    t = ttn[pos]

      
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
    
    # for excited state the action just needs to return the normal version plus the output of inner(ttn_running, ttn_ground, pos)

    o1s = [inner(sp.ttn, ttn_ortho, pos) for ttn_ortho in sp.ttns_ortho]
    action = ∂A(pTPO, o1s, sp.weight, pos)
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


function next_position(sp::ExcitedSweepHandler, cur_pos::Tuple{Int,Int})
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
