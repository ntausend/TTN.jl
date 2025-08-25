mutable struct SimpleSweepHandlerCPU <: AbstractSimpleSweepHandler
    const number_of_sweeps::Int
    ttn::TreeTensorNetwork
    pTPO::ProjTPO_GPU
    func
        
    maxdims::Vector{Int64}

    dir::Symbol
    current_sweep::Int
    current_energy::Float64
    ## only for subspace expansion and noise
    # current_spec::Spectrum
    # current_max_truncerr::Float64
    outputlevel::Int
    # use_gpu::Bool
    SimpleSweepHandlerCPU(ttn, pTPO, func, n_sweeps, maxdims, outputlevel = 0) = 
        new(n_sweeps, ttn, pTPO, func, maxdims, :up, 1, 0., outputlevel)
        # new(n_sweeps, ttn, pTPO, func, maxdims, :up, 1, 0., Spectrum(nothing, 0.0), 0.0, outputlevel, use_gpu)
end


## probably not needed after reset
function initialize!(sp::SimpleSweepHandlerCPU)
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

