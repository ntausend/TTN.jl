# Krylov Parameters, DMRG
global const DEFAULT_TOL_DMRG              = 1e-14
global const DEFAULT_KRYLOVDIM_DMRG        = 5
global const DEFAULT_MAXITER_DMRG          = 3
global const DEFAULT_VERBOSITY_DMRG        = 0
global const DEFAULT_ISHERMITIAN_DMRG      = true
global const DEFAULT_WHICH_EIGENVALUE_DMRG = :SR

# Krylov Parameters, TDVP
global const DEFAULT_TOL_TDVP         = 1e-12
global const DEFAULT_KRYLOVDIM_TDVP   = 30
global const DEFAULT_MAXITER_TDVP     = 3
global const DEFAULT_VERBOSITY_TDVP   = 0
global const DEFAULT_ISHERMITIAN_TDVP = true
global const DEFAULT_EAGER_TDVP       = true

function sweep(psi0::TreeTensorNetwork, sp::AbstractSweepHandler; kwargs...)
    
    obs = get(kwargs, :observer, NoObserver())

    outputlevel = get(kwargs, :outputlevel, 1)
    name_ttn = get(kwargs, :name_ttn, nothing)

    svd_alg = get(kwargs, :svd_alg, nothing)

    # now start with the sweeping protocol
    initialize!(sp)
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

        if !isnothing(name_ttn)
            mode = isfile(name_ttn * ".h5") ? "r+" : "w"
            println("saving at t= $(round(sw,digits=3))")
            h5open(name_ttn*".h5", mode) do file
                write(file, "ttn/t=$(round(sw,digits=3))", cpu(sp.ttn))
            end
        end

        if outputlevel ≥ 1
            print("Finished sweep $sw. ")
            @printf("Needed Time %.3fs\n", t_f - t_p)
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

function dmrg(psi0::TreeTensorNetwork, tpo::AbstractTensorProductOperator; expander = NoExpander(), kwargs...)

    n_sweeps::Int64 = get(kwargs, :number_of_sweeps, 1)
    maxdims::Union{Int64, Vector{Int64}}   = get(kwargs, :maxdims, 1)
    noise::Union{<:Real, Vector{<:Real}} = get(kwargs, :noise, 0.0)

    outputlevel = get(kwargs, :outputlevel, 1)

    if maxdims isa Int64
        maxdims = [maxdims]
    end
    #maxdims = vcat(maxdims, repeat(maxdims[end:end], n_sweeps - length(maxdims)+1))
    if noise isa Float64
        noise = [noise]
    end
    #noise = vcat(abs.(noise), repeat(noise[end:end], n_sweeps - length(noise)+1))

    eigsolve_tol = get(kwargs, :eigsovle_tol, DEFAULT_TOL_DMRG)
    eigsolve_krylovdim = get(kwargs, :eigsovle_krylovdim, DEFAULT_KRYLOVDIM_DMRG)
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_DMRG)
    #eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_DMRG)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_DMRG)
    eigsolve_which_eigenvalue = get(kwargs, :which_eigenvalue, DEFAULT_WHICH_EIGENVALUE_DMRG)

    psic = copy(psi0)
    psic = move_ortho!(psic, (number_of_layers(network(psic)),1))

    pTPO = ProjectedTensorProductOperator(psic, tpo)
    func = (action, T) -> eigsolve(action, T, 1,
                            eigsolve_which_eigenvalue;
                            ishermitian=ishermitian,
                            tol=eigsolve_tol,
                            krylovdim=eigsolve_krylovdim,
                            maxiter=eigsolve_maxiter)

    sh = SimpleSweepHandler(psic, pTPO, func, n_sweeps, maxdims, noise, expander, outputlevel)
    return sweep(psic, sh; kwargs...)
end

function dmrg(psi0::TreeTensorNetwork, psi_ortho::Vector, tpo::AbstractTensorProductOperator; expander = NoExpander(), kwargs...)

    n_sweeps::Int64 = get(kwargs, :number_of_sweeps, 1)
    maxdims::Union{Int64, Vector{Int64}}   = get(kwargs, :maxdims, 1)
    noise::Union{Float64, Vector{Float64}} = get(kwargs, :noise, 0.0)
    weight::Float64 = get(kwargs, :weight, 10.0)
    if_old_excitedSH::Bool = get(kwargs, :if_old_excited, false)

    if maxdims isa Int64
        maxdims = [maxdims]
    end
    #maxdims = vcat(maxdims, repeat(maxdims[end:end], n_sweeps - length(maxdims)+1))
    if noise isa Float64
        noise = [noise]
    end
    #noise = vcat(abs.(noise), repeat(noise[end:end], n_sweeps - length(noise)+1))

    eigsolve_tol = get(kwargs, :eigsovle_tol, DEFAULT_TOL_DMRG)
    eigsolve_krylovdim = get(kwargs, :eigsovle_krylovdim, DEFAULT_KRYLOVDIM_DMRG)
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_DMRG)
    #eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_DMRG)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_DMRG)
    eigsolve_which_eigenvalue = get(kwargs, :which_eigenvalue, DEFAULT_WHICH_EIGENVALUE_DMRG)

    psic = copy(psi0)
    psic = move_ortho!(psic, (number_of_layers(network(psic)),1))

    pTPO = ProjectedTensorProductOperator(psic, tpo)
    func = (action, T) -> eigsolve(action, T, 1,
                            eigsolve_which_eigenvalue;
                            ishermitian=ishermitian,
                            tol=eigsolve_tol,
                            krylovdim=eigsolve_krylovdim,
                            maxiter=eigsolve_maxiter)

    if if_old_excitedSH
        sh = ExcitedSweepHandler(psic, psi_ortho, pTPO, func, n_sweeps, maxdims, noise, expander, weight)
    else
        pTTNs = ProjTTN(psi0, psi_ortho, weight.*ones(length(psi_ortho)))
        full_ptpo = VecProj(tuple(pTPO,pTTNs...))
        sh = SimpleSweepHandler(psic, full_ptpo, func, n_sweeps, maxdims, noise, expander)
    end 
    return sweep(psic, sh; kwargs...)
end

function tdvp(psi0::TreeTensorNetwork, tpo::AbstractTensorProductOperator; kwargs...)
    eigsolve_tol = get(kwargs, :eigsovle_tol, DEFAULT_TOL_TDVP)
    eigsolve_krylovdim = get(kwargs, :eigsovle_krylovdim, DEFAULT_KRYLOVDIM_TDVP)
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_TDVP)
    #eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_TDVP)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_TDVP)
    eigsolve_eager = get(kwargs, :eager, DEFAULT_EAGER_TDVP)

    timestep = get(kwargs, :timestep, 1e-2)
    initialtime = get(kwargs, :initialtime, 0.)
    finaltime = get(kwargs, :finaltime, 1.)
    psic = copy(psi0)
    psic = move_ortho!(psic, (number_of_layers(network(psic)),1))

    pTPO = ProjectedTensorProductOperator(psic, tpo)

    func = (action, dt, T) -> exponentiate(action, convert(eltype(T), -1im*dt), T,
                                           krylovdim = eigsolve_krylovdim,
                                           tol = eigsolve_tol, 
                                           maxiter = eigsolve_maxiter,
                                           ishermitian = ishermitian,
                                           eager = eigsolve_eager);  

    return sweep(psic, TDVPSweepHandler(psic, pTPO, timestep, initialtime, finaltime, func); kwargs...);
end
