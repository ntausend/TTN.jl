mutable struct SimpleSweepHandlerCPU <: AbstractSimpleSweepHandler
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
    function SimpleSweepHandlerCPU(ttn, pTPO, func, n_sweeps, maxdims, outputlevel = 0)
        path = ttn_traversal_least_steps(network(ttn); include_layer0=false, exclude_topnode=false)
        return new(n_sweeps, ttn, pTPO, func, maxdims, :up, path.visit_order, 1, 0., outputlevel)
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

    action = ∂A_GPU(pTPO, pos; use_gpu = false)

    val, tn = sp.func(action, T)
    sp.current_energy = real(val[1])
    tn = tn[1]

    ttn[pos] = tn

    pn = next_position(sp, pos)
    if isnothing(pn) 
        ttn[pos] = tn
        return ttn
    end
    # isnothing(pn) && ttn[pos] = tn && return ttn
    move_ortho!(ttn, pn; normalize = true)

    pTPO = set_position!(pTPO, ttn; use_gpu = false)

    sp.ttn = ttn
    return sp
end

