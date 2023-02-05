# Krylov Parameters, DMRG
global const DEFAULT_TOL_DMRG              = 1e-14
global const DEFAULT_KRYLOVDIM_DMRG        = 5
global const DEFAULT_MAXITER_DMRG          = 3
global const DEFAULT_VERBOSITY_DMRG        = 0
global const DEFAULT_ISHERMITIAN_DMRG      = true
global const DEFAULT_WHICH_EIGENVALUE_DMRG = :SR



global const DEFAULT_TOL_TDVP         = 1e-12
global const DEFAULT_KRYLOVDIM_TDVP   = 30
global const DEFAULT_MAXITER_TDVP     = 3
global const DEFAULT_VERBOSITY_TDVP   = 0
global const DEFAULT_ISHERMITIAN_TDVP = true
global const DEFAULT_EAGER_TDVP       = true

function sweep(psi0::TreeTensorNetwork, sp::AbstractSweepHandler; kwargs...)
    
    obs = get(kwargs, :observer, NoObserver())

    outputlevel = get(kwargs, :outputlevel, 1)

    # now start with the sweeping protocol
    initialize!(sp)
    #sp = SimpleSweepProtocol(net, n_sweeps)
    for sw in sweeps(sp)
        if outputlevel ≥ 2 
            println("Start sweep number $(sw)")
            flush(stdout)
        end
        t_p = time()
        for pos in sp
            update!(sp, pos)
            measure!(
                obs;
                sweep_handler=sp,
                pos=pos,
                outputlevel=outputlevel
            )
        end
        t_f = time()
        if outputlevel ≥ 1
            #println(repeat("=", 50))
            @printf("Finsihed sweep %i. Needed Time: %0.3fs\n", sw, t_f - t_p)
            # additional info string provided by the sweephandler
            info_string(sp, outputlevel)
            @printf("\n")

            #println(repeat("=", 50))
            flush(stdout)
        end
    end
    return sp
end

function dmrg(psi0::TreeTensorNetwork, mpo::MPOWrapper; expander = NoExpander(), kwargs...)

    n_sweeps::Int64 = get(kwargs, :number_of_sweeps, 1)
    maxdims::Union{Int64, Vector{Int64}}  = get(kwargs, :maxdims, 1)

    if maxdims isa Int64
        maxdims = [maxdims]
    end
    maxdims = vcat(maxdims, repeat([maxdims[end]], n_sweeps - length(maxdims)+1))

    eigsolve_tol = get(kwargs, :eigsovle_tol, DEFAULT_TOL_DMRG)
    eigsolve_krylovdim = get(kwargs, :eigsovle_krylovdim, DEFAULT_KRYLOVDIM_DMRG)
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_DMRG)
    #eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_DMRG)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_DMRG)
    eigsolve_which_eigenvalue = get(kwargs, :which_eigenvalue, DEFAULT_WHICH_EIGENVALUE_DMRG)

    psic = copy(psi0)
    psic = move_ortho!(psic, (number_of_layers(network(psic)),1))

    pTPO = ProjMPO(psic, mpo)
    func = (action, T) -> eigsolve(action, T, 1,
                        eigsolve_which_eigenvalue;
                        ishermitian=ishermitian,
                        tol=eigsolve_tol,
                        krylovdim=eigsolve_krylovdim,
                        maxiter=eigsolve_maxiter)
    return sweep(psic, SimpleSweepHandler(psic, pTPO, func, n_sweeps, maxdims, expander); kwargs...)
end

function dmrg(psi0::TreeTensorNetwork, tpo::TPO; expander = NoExpander(), kwargs...)
    n_sweeps::Int64 = get(kwargs, :number_of_sweeps, 1)
    maxdims::Union{Int64, Vector{Int64}}  = get(kwargs, :maxdims, 1)

    eigsolve_tol = get(kwargs, :eigsovle_tol, DEFAULT_TOL_DMRG)
    eigsolve_krylovdim = get(kwargs, :eigsovle_krylovdim, DEFAULT_KRYLOVDIM_DMRG)
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_DMRG)
    #eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_DMRG)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_DMRG)
    eigsolve_which_eigenvalue = get(kwargs, :which_eigenvalue, DEFAULT_WHICH_EIGENVALUE_DMRG)

    if maxdims isa Int64
        maxdims = [maxdims]
    end
    maxdims = vcat(maxdims, repeat([maxdims[end]], n_sweeps - length(maxdims)+1))

    psic = copy(psi0)
    psic = move_ortho!(psic, (number_of_layers(network(psic)),1))

    pTPO = ProjTPO(psic, tpo)
    func = (action, T) -> eigsolve(action, T, 1,
                            eigsolve_which_eigenvalue;
                            ishermitian=ishermitian,
                            tol=eigsolve_tol,
                            krylovdim=eigsolve_krylovdim,
                            maxiter=eigsolve_maxiter)
    return sweep(psic, SimpleSweepHandler(psic, pTPO, func, n_sweeps, maxdims, expander); kwargs...)
end

function tdvp(psi0::TreeTensorNetwork, mpo::MPOWrapper; kwargs...)
    eigsolve_tol = get(kwargs, :eigsovle_tol, DEFAULT_TOL_TDVP)
    eigsolve_krylovdim = get(kwargs, :eigsovle_krylovdim, DEFAULT_KRYLOVDIM_TDVP)
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_TDVP)
    #eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_TDVP)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_TDVP)
    eigsolve_eager = get(kwargs, :eager, DEFAULT_EAGER_TDVP)

    timestep = get(kwargs, :timestep, 1e-2)
    finaltime = get(kwargs, :finaltime, 1.)
    psic = copy(psi0)
    psic = move_ortho!(psic, (number_of_layers(network(psic)),1))

    pTPO = ProjMPO(psic, mpo)
    
    func = (action, dt, T) -> exponentiate(action, convert(eltype(T), -1im*dt), T,
                                           krylovdim = eigsolve_krylovdim,
                                           tol = eigsolve_tol, 
                                           maxiter = eigsolve_maxiter,
                                           ishermitian = ishermitian,
                                           eager = eigsolve_eager);  

    return sweep(psic, TDVPSweepHandler(psic, pTPO, timestep, finaltime, func); kwargs...);
end

function tdvp(psi0::TreeTensorNetwork, tpo::TPO; kwargs...)
    eigsolve_tol = get(kwargs, :eigsovle_tol, DEFAULT_TOL_TDVP)
    eigsolve_krylovdim = get(kwargs, :eigsovle_krylovdim, DEFAULT_KRYLOVDIM_TDVP)
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_TDVP)
    #eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_TDVP)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_TDVP)
    eigsolve_eager = get(kwargs, :eager, DEFAULT_EAGER_TDVP)

    timestep = get(kwargs, :timestep, 1e-2)
    finaltime = get(kwargs, :finaltime, 1.)
    psic = copy(psi0)
    psic = move_ortho!(psic, (number_of_layers(network(psic)),1))

    pTPO = ProjTPO(psic, tpo)
    
    func = (action, dt, T) -> exponentiate(action, convert(eltype(T), -1im*dt), T,
                                           krylovdim = eigsolve_krylovdim,
                                           tol = eigsolve_tol, 
                                           maxiter = eigsolve_maxiter,
                                           ishermitian = ishermitian,
                                           eager = eigsolve_eager);  

    return sweep(psic, TDVPSweepHandler(psic, pTPO, timestep, finaltime, func); kwargs...)
end
