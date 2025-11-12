function dmrg(psi0::TreeTensorNetwork, tpo::TPO_GPU; expander = NoExpander(), kwargs...)

    n_sweeps::Int64 = get(kwargs, :number_of_sweeps, 1)
    maxdims::Union{Int64, Vector{Int64}}   = get(kwargs, :maxdims, 1)
    noise::Union{<:Real, Vector{<:Real}} = get(kwargs, :noise, 0.0)

    outputlevel = get(kwargs, :outputlevel, 1)
    use_gpu = get(kwargs, :use_gpu, false)
    full_krylov = get(kwargs, :full_krylov, true)

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

    if use_gpu
        node_cache = Dict{Tuple{Int,Int}, ITensor}()
        psic = move_ortho!(psic, (1,1), node_cache)
        pTPO = ProjTPO_GPU(tpo, psic; use_gpu = true, node_cache = node_cache)
        
        if full_krylov
            # println("Using full eigensolver on GPU")
            func = (action, T) -> begin
                T_ = gpu(T)
                eigsolve(action, T_, 1, eigsolve_which_eigenvalue;
                    ishermitian=ishermitian,
                    tol=eigsolve_tol,
                    krylovdim=eigsolve_krylovdim,
                    maxiter=eigsolve_maxiter,
                    verbosity=eigsolve_verbosity)
            end
        else
            # println("Using two-pass eigensolver on GPU")
            func = (action, T) -> begin
                T_ = gpu(T)
                eigsolve_twopass(action, T_;
                    howmany=1,
                    which=eigsolve_which_eigenvalue,
                    krylovdim=eigsolve_krylovdim,
                    tol=eigsolve_tol)
            end
        end
        sh = SimpleSweepHandlerGPU(psic, pTPO, func, n_sweeps, maxdims, expander, outputlevel)
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
        sh = SimpleSweepHandlerCPU(psic, pTPO, func, n_sweeps, maxdims, expander, outputlevel)
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
    
    measure!(
        obs;
        sweep_handler=sp,
        outputlevel=outputlevel,
        dt = 0,
    )
    #sp = SimpleSweepProtocol(net, n_sweeps)
    for sw in sweeps(sp)
        if outputlevel ≥ 2 
            println("Start sweep number $(sw)")
            flush(stdout)
        end
        t_p = time()
        for pos in sp
            # println("Updating position: $pos")
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
    
    measure!(
        obs;
        sweep_handler=sp,
        outputlevel=outputlevel,
        dt = 0,
    )
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

function tdvp(psi0::TreeTensorNetwork, tpo::TPO_GPU; kwargs...)
    
    timestep = get(kwargs, :timestep, 1e-2)
    initialtime = get(kwargs, :initialtime, 0.)
    finaltime = get(kwargs, :finaltime, 1.)

    use_gpu = get(kwargs, :use_gpu, false)
    full_krylov = get(kwargs, :full_krylov, true)
    imaginary_time = get(kwargs, :imaginary_time, false)
    energy_shift = get(kwargs, :energy_shift, false)
    
    # Krylov Parameters
    eigsolve_tol = imaginary_time ? get(kwargs, :eigsolve_tol, DEFAULT_TOL_TDVP_IMAGINARY) : get(kwargs, :eigsolve_tol, DEFAULT_TOL_TDVP)
    eigsolve_krylovdim = imaginary_time ? get(kwargs, :eigsolve_krylovdim, DEFAULT_KRYLOVDIM_TDVP_IMAGINARY) : get(kwargs, :eigsolve_krylovdim, DEFAULT_KRYLOVDIM_TDVP)

    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_TDVP)
    # eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_TDVP)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_TDVP)
    eigsolve_eager = get(kwargs, :eager, DEFAULT_EAGER_TDVP)

    # energy_shift can be: nothing (off), a Number (fixed), or :expectation (dynamic ≈ ⟨H⟩)
    # energy_shift        = get(kwargs, :energy_shift, nothing)

    psic = copy(psi0)

    _timeparam = (T, dt) -> begin
        if imaginary_time
            # For imaginary-time, ensure complex element type
            return convert(eltype(T), (-1.0) * dt)
        else
            return convert(eltype(T), (-1im) * dt)
        end
    end

    _shift_action = function (action, T)
        # compute s = <T|H_eff T>/<T|T>
        HT = action(T)
        num = real(dot(T, HT))
        den = real(dot(T, T))
        s = den == 0 ? zero(num) : num/den
        return v -> action(v) - convert(eltype(v), s) * v
    end

    if use_gpu 
        node_cache = Dict{Tuple{Int,Int}, ITensor}()
        psic = move_ortho!(psic, (number_of_layers(network(psic)),1), node_cache)
        pTPO = ProjTPO_GPU(tpo, psic; use_gpu = true, node_cache = node_cache)

        if full_krylov
            # func = (action, dt, T) -> exponentiate(action, _timeparam(T, dt), T,
            #                                     krylovdim = eigsolve_krylovdim,
            #                                     tol = eigsolve_tol, 
            #                                     maxiter = eigsolve_maxiter,
            #                                     ishermitian = ishermitian,
            #                                     eager = eigsolve_eager);
            func = (action, dt, T) -> begin
                energy_shift && (action = _shift_action(action, T))
                exponentiate(action, _timeparam(T, dt), T;
                             krylovdim = eigsolve_krylovdim,
                             tol = eigsolve_tol,
                             maxiter = eigsolve_maxiter,
                             ishermitian = ishermitian,
                             eager = eigsolve_eager)
            end
        else
            # func = (action, dt, T) -> exponentiate_twopass(action, _timeparam(T, dt), T;
            #                                    krylovdim = eigsolve_krylovdim, tol = eigsolve_tol);
            func = (action, dt, T) -> begin
                energy_shift && (action = _shift_action(action, T))
                exponentiate_twopass(action, _timeparam(T, dt), T;
                                     krylovdim = eigsolve_krylovdim, tol = eigsolve_tol)
            end
        end

        sh = TDVPSweepHandlerGPU(psic, pTPO, timestep, initialtime, finaltime, func, imaginary_time)
        return sweep(psic, sh; node_cache = node_cache, kwargs...)
    else 
        psic = move_ortho!(psic, (number_of_layers(network(psic)),1))
        pTPO = ProjTPO_GPU(tpo, psic; use_gpu = false)
        # func = (action, dt, T) -> exponentiate(action, _timeparam(T, dt), T,
        #                                     krylovdim = eigsolve_krylovdim,
        #                                     tol = eigsolve_tol, 
        #                                     maxiter = eigsolve_maxiter,
        #                                     ishermitian = ishermitian,
        #                                     eager = eigsolve_eager);
        func = (action, dt, T) -> begin
            energy_shift && (action = _shift_action(action, T))

            exponentiate(action, _timeparam(T, dt), T;
                             krylovdim = eigsolve_krylovdim,
                             tol = eigsolve_tol,
                             maxiter = eigsolve_maxiter,
                             ishermitian = ishermitian,
                             eager = eigsolve_eager)
        end
        sh = TDVPSweepHandlerCPU(psic, pTPO, timestep, initialtime, finaltime, func, imaginary_time)
        return sweep(psic, sh;kwargs...)
    end
end

function sweep(psi0::TreeTensorNetwork, sp::TDVPSweepHandlerGPU; node_cache::Dict, kwargs...)
    
    obs = get(kwargs, :observer, NoObserver())

    outputlevel = get(kwargs, :outputlevel, 1)

    svd_alg = get(kwargs, :svd_alg, nothing)

    # now start with the sweeping protocol
    measure!(
        obs;
        sweep_handler=sp,
        outputlevel=outputlevel,
        dt = 0.0)
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
            delete!(node_cache, pos)
        end
        CUDA.synchronize()
        
        t_f = time()
        measure!(
            obs;
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
			outputlevel=outputlevel
		)
	    isdone && break
    end
    return sp
end

function sweep(psi0::TreeTensorNetwork, sp::TDVPSweepHandlerCPU; kwargs...)

    obs = get(kwargs, :observer, NoObserver())

    outputlevel = get(kwargs, :outputlevel, 1)

    svd_alg = get(kwargs, :svd_alg, nothing)

    # now start with the sweeping protocol
    measure!(
        obs;
        sweep_handler=sp,
        outputlevel=outputlevel,
        dt = 0.0)

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
        # sp.imaginary_time && normalize!(sp.ttn)
        measure!(
            obs;
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
			outputlevel=outputlevel
		)
	    isdone && break
    end
    return sp
end
