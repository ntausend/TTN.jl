mutable struct SimpleSweepHandlerGPU <: AbstractSimpleSweepHandler
    const number_of_sweeps::Int
    ttn::TreeTensorNetwork
    pTPO::ProjTPO_GPU
    func
        
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
    function SimpleSweepHandlerGPU(ttn, pTPO, func, n_sweeps, maxdims, expander = NoExpander(), outputlevel = 0)
        path = ttn_traversal_least_steps(network(ttn); include_layer0=false, exclude_topnode=false)
        return new(n_sweeps, ttn, pTPO, func, expander, maxdims, :up, path.visit_order, 0, 0., outputlevel)
    end
end

function next_position(sp::SimpleSweepHandlerGPU, cur_pos::Tuple{Int,Int})
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

function update!(sp::SimpleSweepHandlerGPU,
                 pos::Tuple{Int, Int};
                 svd_alg = nothing,
                 node_cache::Dict = Dict())

    @assert pos == ortho_center(sp.ttn)
    ttn = sp.ttn
    pTPO = sp.pTPO

    # pTPO = set_position!(pTPO, ttn; use_gpu = use_gpu, node_cache = node_cache)
    T = haskey(node_cache, pos) ? node_cache[pos] : gpu(ttn[pos])

    # Expansion target
    pn = next_position(sp, pos)
    posnext = (!isnothing(pn) ? connecting_path(network(ttn), pos, pn)[1] : nothing)
    use_expansion = (!isnothing(posnext) && sp.expander !== NoExpander())

    if use_expansion
        #println("Working on expansion at position $pos towards $posnext")

        if length(inds(T)) != 3
            # Expanding parent node using both children
            child_nds = child_nodes(network(ttn), pos)

            # Pull both children on GPU
            T_child1 = get!(node_cache, child_nds[1], gpu(ttn[child_nds[1]]))
            T_child2 = get!(node_cache, child_nds[2], gpu(ttn[child_nds[2]]))

            # combine child1 with original tensor
            T_temp = T * T_child1

            # Expand combined tensor with child2
            T_temp, T_child2 = expand(T_temp, T_child2, sp.expander; reorthogonalize = true)

            # Split back to original tensor and child1
            ids_shared = commonind(T_child1, T)
            ids_linked = uniqueinds(T_child1, ids_shared)
            T_child1, T = factorize(T_temp, ids_linked; tags = tags(ids_shared))

            # Commit expanded tensors: write CPU to TTN, keep GPU in cache
            (ttn[pos], node_cache[pos], ttn[child_nds[2]], node_cache[child_nds[2]], ttn[child_nds[1]], node_cache[child_nds[1]]) = (cpu(T), T, cpu(T_child2), T_child2, cpu(T_child1), T_child1)

            # Update environments for expansion
            recalc_expander_path_flows!(pTPO, ttn, pos, child_nds[2]; use_gpu = true, node_cache = node_cache)
            recalc_expander_path_flows!(pTPO, ttn, pos, child_nds[1]; use_gpu = true, node_cache = node_cache)

            # IDs changed -> reload both & refresh cache
            T      = (node_cache[pos]         = gpu(ttn[pos]))
            T_child2 = (node_cache[child_nds[2]] = gpu(ttn[child_nds[2]]))
            T_child1 = (node_cache[child_nds[1]] = gpu(ttn[child_nds[1]]))

        else
            # Pull neighbor on GPU
            T_next = get!(node_cache, posnext, gpu(ttn[posnext]))

            # Expand both tensors on GPU
            T, T_next = expand(T, T_next, sp.expander; reorthogonalize = true)

            # Commit expanded tensors: write CPU to TTN, keep GPU in cache
            (ttn[pos], node_cache[pos], ttn[posnext], node_cache[posnext]) =(cpu(T), T, cpu(T_next), T_next)

            # Update environments for expansion
	    recalc_expander_path_flows!(pTPO, ttn, pos, posnext; use_gpu = true, node_cache = node_cache)

            # IDs changed -> reload both & refresh cache (single shot)
            T      = (node_cache[pos]     = gpu(ttn[pos]))
            T_next = (node_cache[posnext] = gpu(ttn[posnext]))
        end
    end


    # Optimize current site
    action = ∂A_GPU(pTPO, pos; use_gpu = true)
    val, T_prime = sp.func(action, T)
    sp.current_energy = real(val isa Number ? val : val[1])
    T_prime = T_prime[1]


    # Commit current site
    (ttn[pos], node_cache[pos]) = (cpu(T_prime), T_prime)

    if !use_expansion
        if !isnothing(pn)
            move_ortho!(ttn, pn, node_cache; normalize = true)
        end
        sp.pTPO = set_position!(pTPO, ttn; use_gpu = true, node_cache = node_cache)
        delete!(node_cache, pos)
        sp.ttn = ttn
        return sp
    end

    if use_expansion
        # Prepare for neighbor optimization: move OC and refresh envs
        move_ortho!(ttn, posnext, node_cache; normalize = true)
        recalc_path_flows!(pTPO, ttn, pos, posnext; use_gpu = true, node_cache = node_cache)

        # IDs changed -> reload both in one shot
        T      = (node_cache[pos]     = gpu(ttn[pos]))
        T_next = (node_cache[posnext] = gpu(ttn[posnext]))

        # Optimize neighbor
        action = ∂A_GPU(pTPO, posnext; use_gpu = true)
        val, T_next_prime = sp.func(action, T_next)
        sp.current_energy = real(val isa Number ? val : val[1])
        T_next_prime = T_next_prime[1]

        # Commit neighbor
        (ttn[posnext], node_cache[posnext]) = (cpu(T_next_prime), T_next_prime)


        # Move OC back, refresh envs, then reload both tensors (IDs changed)
        move_ortho!(ttn, pos, node_cache; normalize = true)
        recalc_path_flows!(pTPO, ttn, posnext, pos; use_gpu = true, node_cache = node_cache)

        # Reload both tensors on GPU
        T      = (node_cache[pos]     = gpu(ttn[pos]))
        T_next = (node_cache[posnext] = gpu(ttn[posnext]))

        # Define epsilon: start from the current combined two-site tensor
        combined = T * T_next
        epsilon  = Inf

        # Iterative refinement
        curiter = 1
        while (curiter < maxiter(sp.expander)) && epsilon > tol(sp.expander)

            # Update original
            action = ∂A_GPU(pTPO, pos; use_gpu = true)
            val, T_prime = sp.func(action, T)
            sp.current_energy = real(val isa Number ? val : val[1])
            T_prime = T_prime[1]

            # Commit original
            (ttn[pos], node_cache[pos]) = (cpu(T_prime), T_prime)

            # Move to neighbor and refresh envs -> IDs change
            move_ortho!(ttn, posnext, node_cache; normalize = true)
            recalc_path_flows!(pTPO, ttn, pos, posnext; use_gpu = true, node_cache = node_cache)

            # Reload both updated tensors
            T      = (node_cache[pos]     = gpu(ttn[pos]))
            T_next = (node_cache[posnext] = gpu(ttn[posnext]))

            # Update neighbor
            action = ∂A_GPU(pTPO, posnext; use_gpu = true)
            val, T_next_prime = sp.func(action, T_next)
            sp.current_energy = real(val isa Number ? val : val[1])
            T_next_prime = T_next_prime[1]

            # update epsilon
            new_combined = T * T_next_prime
            epsilon = norm(new_combined - combined)
            combined = new_combined

            # Commit neighbor
            (ttn[posnext], node_cache[posnext]) = (cpu(T_next_prime), T_next_prime)

            # Move back, refresh envs -> IDs change
            move_ortho!(ttn, pos, node_cache; normalize = true)
            recalc_path_flows!(pTPO, ttn, posnext, pos; use_gpu = true, node_cache = node_cache)

            # Reload both again in one shot
            T      = (node_cache[pos]     = gpu(ttn[pos]))
            T_next = (node_cache[posnext] = gpu(ttn[posnext]))

            curiter += 1
        end
        
        #println("Expansion refinement converged in $curiter iterations with ε = $epsilon")
        
        # Cleanup neighbor from cache
        delete!(node_cache, posnext)
    end

    # Advance OC and finalize env position
    if !isnothing(pn)
        move_ortho!(ttn, pn, node_cache; normalize = true)
    end
    sp.pTPO = set_position!(pTPO, ttn; use_gpu = true, node_cache = node_cache)

    # Cleanup current from cache
    delete!(node_cache, pos)

    sp.ttn = ttn
    return sp
end

function update_node_and_move_gpu!(ttn::TreeTensorNetwork, A::ITensor, position_next::Union{Tuple{Int,Int}, Nothing};
                               normalize = nothing,
                               which_decomp = nothing,
                               mindim = nothing,
                               maxdim = nothing,
                               cutoff = nothing,
                               eigen_perturbation = nothing,
                               svd_alg = nothing,
                               use_gpu::Bool = true,
                               node_cache = Dict())

    normalize = replace_nothing(normalize, false)
    @assert is_orthogonalized(ttn)

    pos = ortho_center(ttn)
    if isnothing(position_next)
        # ttn[pos] = use_gpu ? cpu(A) : A
        ttn[pos] = cpu(A)
        return ttn, Spectrum(nothing, 0.0)
    end

    net = network(ttn)
    posnext = connecting_path(net, pos, position_next)[1]
    idx_r = commonind(ttn[pos], ttn[posnext])
    idx_l = uniqueinds(A, idx_r)

    ## should be already on gpu
    if use_gpu
        A_ = haskey(node_cache, pos) ? node_cache[pos] : gpu(ttn[pos])
        tn_next = haskey(node_cache, posnext) ? node_cache[posnext] : gpu(ttn[posnext])
    else
        A_ = A
    end

    tags_r = tags(idx_r)

    if svd_alg == :krylov
        Q, R, spec = factorize_svdsolve(A_, idx_l, maxdim; tags = tags_r)
    else
        Q, R, spec = factorize(A_, idx_l;
                               tags = tags_r,
                               mindim,
                               maxdim,
                               cutoff,
                               which_decomp,
                               eigen_perturbation,
                               svd_alg)
    end

    if use_gpu
        ttn[pos] = cpu(Q)
        node_cache[pos] = Q
        
        tn_next = tn_next * R   # GPU contraction
        normalize && (tn_next ./= norm(tn_next))

        ttn[posnext] = cpu(tn_next)
        node_cache[posnext] = tn_next
    else
        ttn[pos] = Q
        ttn[posnext] = ttn[posnext] * R
        normalize && (ttn[posnext] ./= norm(ttn[posnext]))
    end
   
    ttn.ortho_center .= posnext
    ## move_ortho for longer path?
    return move_ortho!(ttn, position_next), spec
end
