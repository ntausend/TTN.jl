mutable struct SimpleSweepHandlerCPU <: AbstractSimpleSweepHandler
    const number_of_sweeps::Int
    ttn::TreeTensorNetwork
    pTPO::ProjTPO_GPU
    func
    expander::AbstractSubspaceExpander
        
    maxdims::Vector{Int64}

    dir::Symbol
    path::Vector{Tuple{Int,Int}}
    current_sweep::Int
    current_energy::Float64
    ## only for subspace expansion and noise
    # current_spec::Spectrum
    # current_max_truncerr::Float64
    outputlevel::Int
    # use_gpu::Bool
    function SimpleSweepHandlerCPU(ttn, pTPO, func, n_sweeps, maxdims, expander = NoExpander(), outputlevel = 0)
        path = ttn_traversal_least_steps(network(ttn); include_layer0=false, exclude_topnode=false)
        return new(n_sweeps, ttn, pTPO, func, expander, maxdims, :up, path.visit_order, 0, 0., outputlevel)
    end
end

function next_position(sp::SimpleSweepHandlerCPU, cur_pos::Tuple{Int,Int})
    path = sp.path
    idx = findfirst(==(cur_pos), path)

    if sp.dir == :up
        if idx == length(path)
            sp.dir = :down
            return path[idx - 1]
        else
            return path[idx + 1]
        end
    elseif sp.dir == :down
        if idx == 1
            return nothing
        else
            return path[idx - 1]
        end
    end
    error("Invalid direction of the iterator: $(sp.dir)")
end


function update!(sp::SimpleSweepHandlerCPU,
                 pos::Tuple{Int, Int};
                 svd_alg = nothing)

    @assert pos == ortho_center(sp.ttn)
    ttn = sp.ttn
    pTPO = sp.pTPO
    # pTPO = set_position!(pTPO, ttn; use_gpu = use_gpu, node_cache = node_cache)
    T = ttn[pos]


    # Expansion target
    pn = next_position(sp, pos)
    posnext = (!isnothing(pn) ? connecting_path(network(ttn), pos, pn)[1] : nothing)
    use_expansion = (!isnothing(posnext) && sp.expander !== NoExpander())

    if use_expansion
        #println("Working on expansion at position $pos towards $posnext")

        if length(inds(T)) != 3
            # Expanding parent node using both children
            child_nds = child_nodes(network(ttn), pos)

            # Pull both children
            T_child1 = ttn[child_nds[1]]
            T_child2 = ttn[child_nds[2]]

            # combine child1 with original tensor
            T_temp = T * T_child1

            # Expand combined tensor with child2
            T_temp, T_child2 = expand(T_temp, T_child2, sp.expander; reorthogonalize = true)

            # Split back to original tensor and child1
            ids_shared = commonind(T_child1, T)
            ids_linked = uniqueinds(T_child1, ids_shared)
            T_child1, T = factorize(T_temp, ids_linked; tags = tags(ids_shared))

            # Commit expanded tensors: write CPU to TTN
            (ttn[pos], ttn[child_nds[2]], ttn[child_nds[1]]) = (T, T_child2, T_child1)

            # Update environments for expansion
            recalc_expander_path_flows!(pTPO, ttn, pos, child_nds[2]; use_gpu = false)
            recalc_expander_path_flows!(pTPO, ttn, pos, child_nds[1]; use_gpu = false)

            # IDs changed -> reload both
            T      = ttn[pos]
            T_child2 = ttn[child_nds[2]]
            T_child1 = ttn[child_nds[1]]

        else
            # Pull neighbor
            T_next = ttn[posnext]

            # Expand both tensors
            T, T_next = expand(T, T_next, sp.expander; reorthogonalize = true)

            # Commit expanded tensors: write CPU to TTN
            (ttn[pos], ttn[posnext]) =(T, T_next)

            # Update environments for expansion
	        recalc_expander_path_flows!(pTPO, ttn, pos, posnext; use_gpu = false)

            # IDs changed -> reload both & refresh cache (single shot)
            T      = ttn[pos]
            T_next = ttn[posnext]
        end
    end

    action = ∂A_GPU(pTPO, pos; use_gpu = false)

    val, tn = sp.func(action, T)
    sp.current_energy = real(val[1])
    tn = tn[1]

    ttn[pos] = tn

    if !use_expansion
        if !isnothing(pn)
            move_ortho!(ttn, pn; normalize = true)
        end
        sp.pTPO = set_position!(pTPO, ttn; use_gpu = false)
        sp.ttn = ttn
        return sp
    end

    if use_expansion
        # Prepare for neighbor optimization: move OC and refresh envs
        move_ortho!(ttn, posnext; normalize = true)
        recalc_path_flows!(pTPO, ttn, pos, posnext; use_gpu = false)

        # IDs changed -> reload both in one shot
        T      = ttn[pos]
        T_next = ttn[posnext]

        # Optimize neighbor
        action = ∂A_GPU(pTPO, posnext; use_gpu = false)
        val, T_next_prime = sp.func(action, T_next)
        sp.current_energy = real(val isa Number ? val : val[1])
        T_next_prime = T_next_prime[1]

        # Commit neighbor
        ttn[posnext] = T_next_prime


        # Move OC back, refresh envs, then reload both tensors (IDs changed)
        move_ortho!(ttn, pos; normalize = true)
        recalc_path_flows!(pTPO, ttn, posnext, pos; use_gpu = false)

        # Reload both tensors on GPU
        T      = ttn[pos]
        T_next = ttn[posnext]

        # Define epsilon: start from the current combined two-site tensor
        combined = T * T_next
        epsilon  = Inf

        # Iterative refinement
        curiter = 1
        while (curiter < maxiter(sp.expander)) && epsilon > tol(sp.expander)

            # Update original
            action = ∂A_GPU(pTPO, pos; use_gpu = false)
            val, T_prime = sp.func(action, T)
            sp.current_energy = real(val isa Number ? val : val[1])
            T_prime = T_prime[1]

            # Commit original
            ttn[pos] = T_prime

            # Move to neighbor and refresh envs -> IDs change
            move_ortho!(ttn, posnext; normalize = true)
            recalc_path_flows!(pTPO, ttn, pos, posnext; use_gpu = false)

            # Reload both updated tensors
            T      = ttn[pos]
            T_next = ttn[posnext]

            # Update neighbor
            action = ∂A_GPU(pTPO, posnext; use_gpu = false)
            val, T_next_prime = sp.func(action, T_next)
            sp.current_energy = real(val isa Number ? val : val[1])
            T_next_prime = T_next_prime[1]

            # update epsilon
            new_combined = T * T_next_prime
            epsilon = norm(new_combined - combined)
            combined = new_combined

            # Commit neighbor
            ttn[posnext] = T_next_prime

            # Move back, refresh envs -> IDs change
            move_ortho!(ttn, pos; normalize = true)
            recalc_path_flows!(pTPO, ttn, posnext, pos; use_gpu = false)
            # Reload both again in one shot
            T      = ttn[pos]
            T_next = ttn[posnext]

            curiter += 1
        end
        
        #println("Expansion refinement converged in $curiter iterations with ε = $epsilon")
        
    end

    # Advance OC and finalize env position
    if !isnothing(pn)
        move_ortho!(ttn, pn; normalize = true)
    end
    sp.pTPO = set_position!(pTPO, ttn; use_gpu = false)

    sp.ttn = ttn
    return sp
end

