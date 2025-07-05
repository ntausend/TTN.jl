function dmrg(psi0::TreeTensorNetwork, tpo::TPO_GPU; expander = NoExpander(), kwargs...)

    n_sweeps::Int64 = get(kwargs, :number_of_sweeps, 1)
    maxdims::Union{Int64, Vector{Int64}}   = get(kwargs, :maxdims, 1)
    noise::Union{<:Real, Vector{<:Real}} = get(kwargs, :noise, 0.0)

    outputlevel = get(kwargs, :outputlevel, 1)
    use_gpu = get(kwargs, :use_gpu, false)

    if maxdims isa Int64
        maxdims = [maxdims]
    end
    #maxdims = vcat(maxdims, repeat(maxdims[end:end], n_sweeps - length(maxdims)+1))

    eigsolve_tol = get(kwargs, :eigsolve_tol, DEFAULT_TOL_DMRG)
    eigsolve_krylovdim = get(kwargs, :eigsolve_krylovdim, DEFAULT_KRYLOVDIM_DMRG)
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_DMRG)
    eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_DMRG)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_DMRG)
    eigsolve_which_eigenvalue = get(kwargs, :which_eigenvalue, DEFAULT_WHICH_EIGENVALUE_DMRG)

    psic = copy(psi0)
    ## later: move_ortho to first tensor of sweep order
    ## sweep_order = collect(TTN.NodeIterator(net))
    ## sweep_order[1]
    psic = move_ortho!(psic, (1,1))

    pTPO = ProjTPO_GPU(tpo, psic; use_gpu = use_gpu)
    func = (action, T) -> begin
        T_ = use_gpu ? gpu(T) : T
        eigsolve(action, T_, 1, eigsolve_which_eigenvalue;
             ishermitian=ishermitian,
             tol=eigsolve_tol,
             krylovdim=eigsolve_krylovdim,
             maxiter=eigsolve_maxiter,
             verbosity=eigsolve_verbosity)
    end

    sh = SimpleSweepHandlerGPU(psic, pTPO, func, n_sweeps, maxdims, outputlevel, use_gpu)
    return sweep(psic, sh; kwargs...)
end

function sweep(psi0::TreeTensorNetwork, sp::SimpleSweepHandlerGPU; kwargs...)
    
    obs = get(kwargs, :observer, NoObserver())

    outputlevel = get(kwargs, :outputlevel, 1)

    svd_alg = get(kwargs, :svd_alg, nothing)

    # now start with the sweeping protocol
    # already initialised to oc = (1,1) due to 
    # initialize!(sp)
    
    # measure!(
    #     obs;
    #     sweep_handler=sp,
    #     outputlevel=outputlevel,
    #     dt = 0,
    # )
    #sp = SimpleSweepProtocol(net, n_sweeps)
    for sw in sweeps(sp)
        if outputlevel ≥ 2 
            println("Start sweep number $(sw)")
            flush(stdout)
        end
        t_p = time()
        for pos in sp
            update!(sp, pos; svd_alg)
        end
        t_f = time()
        measure!(
            obs;
            sweep_handler=sp,
            outputlevel=outputlevel,
            dt = t_f-t_p,
        )
        if outputlevel ≥ 1
            print("Finished sweep $sw. ")
            @printf("Needed Time %.3fs\n", t_f - t_p)
            # additional info string provided by the sweephandler
            info_string(sp, outputlevel)
            @printf("\n")
            flush(stdout)
        end
    isdone = checkdone!(
			obs;
			sweep_handler=sp,
			outputlevel=outputlevel
		)
	isdone && break
    end
    return sp
end
