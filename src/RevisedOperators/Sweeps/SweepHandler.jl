mutable struct SimpleSweepHandlerGPU <: AbstractRegularSweepHandler
    const number_of_sweeps::Int
    ttn::TreeTensorNetwork
    pTPO::ProjTPO_GPU
    func
    # expander::AbstractSubspaceExpander
        
    maxdims::Vector{Int64}
    # noise::Vector{Number}

    dir::Symbol
    current_sweep::Int
    current_energy::Float64
    current_spec::Spectrum
    current_max_truncerr::Float64
    outputlevel::Int
    use_gpu::Bool
    SimpleSweepHandlerGPU(ttn, pTPO, func, n_sweeps, maxdims, outputlevel = 0, use_gpu = false) = 
        new(n_sweeps, ttn, pTPO, func, maxdims, :up, 1, 0., Spectrum(nothing, 0.0), 0.0, outputlevel, use_gpu)
end


current_sweep(sh::SimpleSweepHandlerGPU) = sh.current_sweep

function maxdim(sh::SimpleSweepHandlerGPU)
    length(sh.maxdims) < current_sweep(sh) && return sh.maxdims[end]
    return sh.maxdims[current_sweep(sh)]
end

function info_string(sh::SimpleSweepHandlerGPU, output_level::Int)
    e = sh.current_energy
    trnc_wght = sh.current_max_truncerr
    # todo ->  make a function for that .... which also can handle TensorKit
    maxdim = maxlinkdim(sh.ttn)
    output_level ≥ 1 && @printf("\tCurrent energy: %.15f.\n", e)
    output_level ≥ 2 && @printf("\tTruncated Weigth: %.3e. Maximal bond dim = %i\n", trnc_wght, maxdim)
    sh.current_max_truncerr = 0.0
    nothing
end

## probably not needed after reset
function initialize!(sp::SimpleSweepHandlerGPU)
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
function update_next_sweep!(sp::SimpleSweepHandlerGPU)
    sp.dir = :up
    sp.current_sweep += 1 
    return sp
end
#=
function update!(sp::SimpleSweepHandlerGPU, pos::Tuple{Int, Int}; svd_alg = nothing)
    @assert pos == ortho_center(sp.ttn)
    ttn = sp.ttn
    pTPO = sp.pTPO
    
    net = network(ttn)

    # adjust the position of the projected operator, i.e. redefine the environemnts
    pTPO = set_position!(pTPO, ttn)

    t = ttn[pos]
      
    pn = next_position(sp,pos)
    # if !isnothing(pn)
    #     recalc_path_flows!(pTPO, ttn, pn)
    # end
    
    action  = ∂A(pTPO, pos)
    val, tn = sp.func(action, t)
    sp.current_energy = real(val[1])
    tn = tn[1]
    
    ttn[pos] = tn

    # possible other arguments, which_decomp, svd_alg etc
    # update the node and shift ttn.oc to next position
    ttn, spec = update_node_and_move!(ttn,tn, pn;
                                      maxdim = maxdim(sp),
                                      normalize = true,
                                      svd_alg)

    sp.current_spec = spec
    trncerr = truncerror(spec)
    sp.current_max_truncerr = max(sp.current_max_truncerr, trncerr)
    return sp
end
=#
function update!(sp::SimpleSweepHandlerGPU,
                 pos::Tuple{Int, Int};
                 svd_alg = nothing)
    @assert pos == ortho_center(sp.ttn)
    ttn = sp.ttn
    pTPO = sp.pTPO

    pTPO = set_position!(pTPO, ttn)

    T = ttn[pos]
    # loading to GPU already done in func definition via dmrg call
    # T_ = use_gpu ? gpu(T) : T
    action = ∂A_GPU(pTPO, pos; use_gpu = sp.use_gpu)

    val, tn = sp.func(action, T)
    sp.current_energy = real(val[1])
    tn = tn[1]

    # always store result on CPU
    ttn[pos] = cpu(tn)

    pn = next_position(sp, pos)
    ttn, spec = update_node_and_move_gpu!(ttn, tn, pn;
                                      maxdim=maxdim(sp),
                                      normalize=true,
                                      svd_alg, use_gpu = sp.use_gpu)

    # ttn, spec = update_node_and_move_gpu!(ttn, ttn[pos], pn;
    #                                   maxdim=maxdim(sp),
    #                                   normalize=true,
    #                                   svd_alg, use_gpu = sp.use_gpu)

    sp.ttn = ttn
    sp.current_spec = spec
    sp.current_max_truncerr = max(sp.current_max_truncerr, truncerror(spec))
    return sp
end



function next_position(sp::SimpleSweepHandlerGPU, cur_pos::Tuple{Int,Int})
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


function update_node_and_move_gpu!(ttn::TreeTensorNetwork, A::ITensor, position_next::Union{Tuple{Int,Int}, Nothing};
                               normalize = nothing,
                               which_decomp = nothing,
                               mindim = nothing,
                               maxdim = nothing,
                               cutoff = nothing,
                               eigen_perturbation = nothing,
                               svd_alg = nothing,
                               use_gpu::Bool = false)

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
    A_ = use_gpu ? gpu(A) : A
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
        Q_ = cpu(Q)  # store Q on CPU
        R_ = R       # keep R on GPU
        ttn[posnext] = cpu(gpu(ttn[posnext]) * R_)  # GPU contraction
    else
        Q_ = Q
        ttn[posnext] = ttn[posnext] * R
    end
    ttn[pos] = Q_

    normalize && (ttn[posnext] ./= norm(ttn[posnext]))
    ttn.ortho_center .= posnext
    ## move_ortho for longer path?
    return move_ortho!(ttn, position_next), spec
end