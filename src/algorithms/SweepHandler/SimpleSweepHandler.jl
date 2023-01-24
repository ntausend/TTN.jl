mutable struct SimpleSweepHandler <: AbstractRegularSweepHandler
    const number_of_sweeps::Int
    ttn::TreeTensorNetwork
    pTPO::ProjTensorProductOperator
    func
    expander::AbstractSubspaceExpander
        
    maxdims::Vector{Int64}
    dir::Symbol
    current_sweep::Int
    energies::Vector{Float64}
    curtime::Float64
    timings::Vector{Float64}
    add_timings::Vector{Float64}
    SimpleSweepHandler(ttn, pTPO, func, n_sweeps, maxdims, expander = NoExpander()) = 
        new(n_sweeps, ttn, pTPO, func, expander, maxdims, :up, 1, Float64[], 0.0, Float64[], Float64[])
end

function initialize!(sp::SimpleSweepHandler)
    ttn = sp.ttn
    pTPO = sp.pTPO

    net = network(ttn)

    #adjust the tree dimension to the first bond dimension
    oc = ortho_center(ttn)
    
    ttn = adjust_tree_tensor_dimensions!(ttn, sp.maxdims[sp.current_sweep]; reorthogonalize = false)

    # check if ttn was altered, this  can be seen by haveing new oc
    if ortho_center(ttn) != oc
        # reorthogonalize
        ttn = move_ortho!(_reorthogonalize!(ttn), (1,1))
        # rebuild the environments
        pTPO = rebuild_environments!(pTPO, ttn)
    else
        # now move everything to the starting point
        ttn = move_ortho!(ttn, (1,1))
        # update the environments accordingly
        pth = connecting_path(net, (number_of_layers(net),1), (1,1))
        pth = vcat((number_of_layers(net),1), pth)
        for (jj,p) in enumerate(pth[1:end-1])
            ism = ttn[p]
            pTPO = update_environments!(pTPO, ism, p, pth[jj+1])
        end
    end
    #@show sp.ttn[number_of_layers(net), 1] == ttn[number_of_layers(net),1]
    #@show inds(ttn[number_of_layers(net), 1])

    sp.ttn = ttn
    sp.pTPO = pTPO
    sp.curtime = time()
    return sp
end

# simple reset the sweep Handler and update the current sweep number
# current number still needed?
function update_next_sweep!(sp::SimpleSweepHandler)
    tn = time()
    push!(sp.timings, tn - sp.curtime)
    sp.curtime = tn
    @printf("Finished Sweep %i. Energy: %.3f. Needed Time: %.3f s\n", sp.current_sweep, sp.energies[end], sp.timings[end])
    flush(stdout)

    sp.dir = :up
    sp.current_sweep += 1 
    return sp
end

function update!(sp::SimpleSweepHandler, pos::Tuple{Int, Int})
    @assert pos == ortho_center(sp.ttn)
    ttn = sp.ttn
    pTPO = sp.pTPO
    
    net = network(ttn)

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
            pTPO = update_environments!(pTPO, Aprime, posnext, pos)
        else
            # top node, here we need some different kind of expansion, depending on the tree type.
            # binary tree needs to enlarge both child indices at once, larger trees not necessary (is this true?)
            chldnds = child_nodes(net, pos)
            A_chlds = map(p -> getindex(ttn, p), chldnds)
            length(A_chlds) == 2 || error("Top node supspace expansion not implemented for non binary trees")
            t, Al, Ar = expand(t, Tuple(A_chlds), sp.expander; reorthogonalize = true)
            ttn[pos] = t
            ttn[chldnds[1]] = Al
            ttn[chldnds[2]] = Ar

            pTPO = update_environments!(pTPO, Al, chldnds[1], pos)
            pTPO = update_environments!(pTPO, Ar, chldnds[2], pos)
        end
    end

    action  = ∂A(pTPO, pos)
    val, tn = sp.func(action, t)
    #@printf("Needed time for minimzation: %.3f\n", t2 - t1)
    push!(sp.energies, real(val[1]))
    tn = tn[1]
    #@show inds(tn)
    # truncate the bond
    ttn = truncate_and_move!(ttn, tn, pos, pn, sp.expander; maxdim = sp.maxdims[sp.current_sweep])
    #; maxdim = sp.maxdims[sp.current_sweep])#, mindim = 1, cutoff = 1E-13)

    
    if !isnothing(pn)
        pth = vcat(pos, pth)
        for (jj,pk) in enumerate(pth[1:end-1])
            ism = ttn[pk]
            pTPO = update_environments!(pTPO, ism, pk, pth[jj+1])
        end
    end
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