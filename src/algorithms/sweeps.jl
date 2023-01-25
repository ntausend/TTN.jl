function sweep(psi0::TreeTensorNetwork, sp::AbstractSweepHandler; kwargs...)
    verbose_level = get(kwargs, :verbose_level, 1)

    # now start with the sweeping protocol
    initialize!(sp)
    #sp = SimpleSweepProtocol(net, n_sweeps)
    for sw in sweeps(sp)
        if verbose_level ≥ 2 
            println("Start sweep number $(sw)")
            flush(stdout)
        end
        for pos in sp
            update!(sp, pos)
        end
        if verbose_level ≥ 1
            println(repeat("=", 50))
            println("Finsihed sweep $(sw).")
            println(repeat("=", 50))
            flush(stdout)
        end
    end
    return sp
end

function dmrg(psi0::TreeTensorNetwork, mpo::AbstractTensorProductOperator; expander = NoExpander(), kwargs...)
    n_sweeps::Int64 = get(kwargs, :number_of_sweeps, 1)
    maxdims::Union{Int64, Vector{Int64}}  = get(kwargs, :maxdims, 1)

    if maxdims isa Int64
        maxdims = [maxdims]
    end
    maxdims = vcat(maxdims, repeat([maxdims[end]], n_sweeps - length(maxdims)+1))

    psic = copy(psi0)
    psic = move_ortho!(psic, (number_of_layers(network(psic)),1))

    pTPO = ProjTensorProductOperator(psic, mpo)
    func = (action, T) -> eigsolve(action, T, 1,
                        eigsolve_which_eigenvalue;
                        ishermitian=ishermitian,
                        tol=eigsolve_tol,
                        krylovdim=eigsolve_krylovdim,
                        maxiter=eigsolve_maxiter)
    return sweep(psic, SimpleSweepHandler(psic, pTPO, func, n_sweeps, maxdims, expander); kwargs...)
end

function tdvp(psi0::TreeTensorNetwork, mpo::AbstractTensorProductOperator; kwargs...)
    timestep = get(kwargs, :timestep, 1e-2)
    finaltime = get(kwargs, :finaltime, 1.)
    psic = copy(psi0)
    psic = move_ortho!(psic, (number_of_layers(network(psic)),1))

    pTPO = ProjTensorProductOperator(psic, mpo)
    func = (action, dt, T) -> exponentiate(action, -1im*dt, T)
    return sweep(psic, TDVPSweepHandler(psic, pTPO, timestep, finaltime, func); kwargs...)
end

# get next step
#if verbose_level ≥ 3
#    println("\tFinished optimizing position $(pos)")
#    flush(stdout)
#end

#=
if !isnothing(pn)
    if verbose_level ≥ 3
        println("\tStart updating environments.")
        flush(stdout)
    end

    move_ortho!(psic, pn)
    pth = connecting_path(net, pos, pn)
    pth = vcat(pos, pth)
    for (jj,pk) in enumerate(pth[1:end-1])
        ism = psic[pk]
        pTPO = update_environments!(pTPO, ism, pk, pth[jj+1])
    end
    if verbose_level ≥ 3
        println("\tFinished updating environments.")
        flush(stdout)
    end
end
            if verbose_level ≥ 3
                println("\tOptimizing position $(pos)")
                flush(stdout)
            end

=#