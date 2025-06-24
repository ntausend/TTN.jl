function dmrg(psi0::TreeTensorNetwork, tpo::TPO_group; expander = NoExpander(), kwargs...)

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

    eigsolve_tol = get(kwargs, :eigsolve_tol, DEFAULT_TOL_DMRG)
    eigsolve_krylovdim = get(kwargs, :eigsolve_krylovdim, DEFAULT_KRYLOVDIM_DMRG)
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_DMRG)
    eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_DMRG)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_DMRG)
    eigsolve_which_eigenvalue = get(kwargs, :which_eigenvalue, DEFAULT_WHICH_EIGENVALUE_DMRG)

    psic = copy(psi0)
    # set orthocenter to (1,1) to start
    # later: set to first element of sweep path
    psic = move_ortho!(psic, (1,1))

    pTPO = ProjTPO_group(tpo, psi0)
    func = (action, T) -> eigsolve(action, T, 1,
                            eigsolve_which_eigenvalue;
                            ishermitian=ishermitian,
                            tol=eigsolve_tol,
                            krylovdim=eigsolve_krylovdim,
                            maxiter=eigsolve_maxiter,
			    verbosity=eigsolve_verbosity)

    sh = SimpleSweepHandler(psic, pTPO, func, n_sweeps, maxdims, noise, expander, outputlevel)
    return sweep(psic, sh; kwargs...)
end