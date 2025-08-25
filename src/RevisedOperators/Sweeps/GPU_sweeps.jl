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
    ## move_ortho! and inner functions have to be adapted for GPU

    if use_gpu
        node_cache = Dict{Tuple{Int,Int}, ITensor}()
        psic = move_ortho!(psic, (1,1), node_cache)

        pTPO = ProjTPO_GPU(tpo, psic; use_gpu = true, node_cache = node_cache)
        func = (action, T) -> begin
            T_ = gpu(T)
            eigsolve(action, T_, 1, eigsolve_which_eigenvalue;
                ishermitian=ishermitian,
                tol=eigsolve_tol,
                krylovdim=eigsolve_krylovdim,
                maxiter=eigsolve_maxiter,
                verbosity=eigsolve_verbosity)
        end

        sh = SimpleSweepHandlerGPU(psic, pTPO, func, n_sweeps, maxdims, outputlevel)
        return sweep(psic, sh; node_cache = node_cache, kwargs...)
    else
        psic = move_ortho!(psic, (1,1))

        pTPO = ProjTPO_GPU(tpo, psic; use_gpu = false)
        func = (action, T) -> begin
            eigsolve(action, T, 1, eigsolve_which_eigenvalue;
                ishermitian=ishermitian,
                tol=eigsolve_tol,
                krylovdim=eigsolve_krylovdim,
                maxiter=eigsolve_maxiter,
                verbosity=eigsolve_verbosity)
        end
        sh = SimpleSweepHandlerCPU(psic, pTPO, func, n_sweeps, maxdims, outputlevel)
        return sweep(psic, sh; kwargs...)
    end    
end

function sweep(psi0::TreeTensorNetwork, sp::SimpleSweepHandlerGPU; node_cache = Dict(), kwargs...)
    
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
            if !haskey(node_cache, pos)
                node_cache[pos] = gpu(psi0[pos])
            end
            update!(sp, pos; svd_alg, node_cache = node_cache)
            haskey(node_cache, pos) && delete!(node_cache, pos)
        end
        t_f = time()
        measure!(obs;
            sweep_handler=sp,
            outputlevel=outputlevel,
            dt = t_f-t_p)
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
			outputlevel=outputlevel,
            sweep_number = sw
		)
	    isdone && break
    end
    return sp
end

function sweep(psi0::TreeTensorNetwork, sp::SimpleSweepHandlerCPU; kwargs...)
    
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
            dt = t_f-t_p
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
			outputlevel=outputlevel,
            sweep_number = sw
		)
	    isdone && break
    end
    return sp
end

function tdvp(psi0::TreeTensorNetwork, tpo::TPO_GPU; full_krylov=false, kwargs...)
    eigsolve_tol = get(kwargs, :eigsolve_tol, DEFAULT_TOL_TDVP)
    eigsolve_krylovdim = get(kwargs, :eigsolve_krylovdim, DEFAULT_KRYLOVDIM_TDVP)
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_TDVP)
    #eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_TDVP)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_TDVP)
    eigsolve_eager = get(kwargs, :eager, DEFAULT_EAGER_TDVP)

    timestep = get(kwargs, :timestep, 1e-2)
    initialtime = get(kwargs, :initialtime, 0.)
    finaltime = get(kwargs, :finaltime, 1.)

    use_gpu = get(kwargs, :use_gpu, false)

    psic = copy(psi0)

    if use_gpu 
        node_cache = Dict{Tuple{Int,Int}, ITensor}()
        psic = move_ortho!(psic, (number_of_layers(network(psic)),1), node_cache)
        pTPO = ProjTPO_GPU(tpo, psic; use_gpu = true, node_cache = node_cache)
        if full_krylov
            func = (action, dt, T) -> exponentiate(action, convert(eltype(T), -1im*dt), T,
                                                krylovdim = eigsolve_krylovdim,
                                                tol = eigsolve_tol, 
                                                maxiter = eigsolve_maxiter,
                                                ishermitian = ishermitian,
                                                eager = eigsolve_eager);
        else
            func = (action, dt, T) -> exponentiate_twopass(action, convert(eltype(T), -1im*dt), T;
                                               krylovdim = eigsolve_krylovdim, tol = eigsolve_tol);
        end

        sh = TDVPSweepHandlerGPU(psic, pTPO, timestep, initialtime, finaltime, func)
        return sweep(psic, sh; node_cache = node_cache, kwargs...)
    else 
        psic = move_ortho!(psic, (number_of_layers(network(psic)),1))
        pTPO = ProjTPO_GPU(tpo, psic; use_gpu = false)
        func = (action, dt, T) -> exponentiate(action, convert(eltype(T), -1im*dt), T,
                                            krylovdim = eigsolve_krylovdim,
                                            tol = eigsolve_tol, 
                                            maxiter = eigsolve_maxiter,
                                            ishermitian = ishermitian,
                                            eager = eigsolve_eager);  
        sh = TDVPSweepHandlerCPU(psic, pTPO, timestep, initialtime, finaltime, func)
        return sweep(psic, sh;kwargs...)
    end
end

function sweep(psi0::TreeTensorNetwork, sp::TDVPSweepHandlerGPU; node_cache::Dict, kwargs...)
    
    obs = get(kwargs, :observer, NoObserver())

    outputlevel = get(kwargs, :outputlevel, 1)

    svd_alg = get(kwargs, :svd_alg, nothing)

    # now start with the sweeping protocol
    # CUDA.memory_status()
    for sw in sweeps(sp)
        if outputlevel ≥ 2 
            println("Start sweep number $(sw)")
            flush(stdout)
            # CUDA.memory_status()
        end
        t_p = time()
        for pos in sp
            if !haskey(node_cache, pos)
                node_cache[pos] = gpu(psi0[pos])
            end
            update!(sp, pos; svd_alg, node_cache = node_cache)
            delete!(node_cache, pos)
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
        # GC.gc()
    end
    # CUDA.memory_status()
    return sp
end

function sweep(psi0::TreeTensorNetwork, sp::TDVPSweepHandlerCPU; kwargs...)

    obs = get(kwargs, :observer, NoObserver())

    outputlevel = get(kwargs, :outputlevel, 1)

    svd_alg = get(kwargs, :svd_alg, nothing)

    # now start with the sweeping protocol 

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
