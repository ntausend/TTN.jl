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
    function SimpleSweepHandlerGPU(ttn, pTPO, func, n_sweeps, maxdims, outputlevel = 0)
        path = ttn_traversal_least_steps(network(ttn); include_layer0=false, exclude_topnode=false)
        return new(n_sweeps, ttn, pTPO, func, maxdims, :up, path.visit_order, 1, 0., outputlevel)
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

    # loading to GPU already done in func definition via dmrg call
    # T_ = use_gpu ? gpu(T) : T
    action = ∂A_GPU(pTPO, pos; use_gpu = true)

    val, tn = sp.func(action, T)
    sp.current_energy = real(val[1])
    tn = tn[1]

    ttn[pos] = cpu(tn)
    node_cache[pos] = tn

    pn = next_position(sp, pos)
    if isnothing(pn)
        ttn[pos] = cpu(tn)
        node_cache[pos] = tn
        return ttn
    end
    move_ortho!(ttn, pn, node_cache; normalize = true)

    pTPO = set_position!(pTPO, ttn; use_gpu = true, node_cache = node_cache)

    delete!(node_cache, pos)
    
    ## needed for truncation after SubspaceExpansion or noise
    #=
    ttn, spec = update_node_and_move_gpu!(ttn, ttn[pos], pn;
                                      maxdim=maxdim(sp),
                                      normalize=true,
                                      svd_alg, use_gpu = sp.use_gpu, node_cache = node_cache)
    
    sp.current_spec = spec
    sp.current_max_truncerr = max(sp.current_max_truncerr, truncerror(spec))
    =#

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