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

    svd_alg = get(kwargs, :svd_alg, nothing)

    # now start with the sweeping protocol
    initialize!(sp)
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
            # measure!(
            #     obs;
            #     sweep_handler=sp,
            #     pos=pos,
            #     outputlevel=outputlevel
            # )
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

function dmrg(psic::TreeTensorNetwork, tpo::AbstractTensorProductOperator; expander = NoExpander(), kwargs...)

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

    # psic = copy(psi0)
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

"""
```julia
   dmrg(psi0::TreeTensorNetwork, psi_ortho::Vector, tpo::AbstractTensorProductOperator; expander = NoExpander(), kwargs...)
```

Performs a dmrg minimization of the initial guess `psi0` with respect to the local Hamiltonian defined by `tpo`, which can either be a MPOWrapper or a TPO object.
`psi_ortho` are additional tensor to orthogonalize against.

It returns a sweep object `sp` where one can extract the final tree with `sp.ttn` and the energy by `sp.current_energy`.

# Keywords:

- `expander`: A optional subspace expansion algorithm can be choosen by this keyword. Be careful as the subspace expansion might be expensive. Default: NoExpander. Possible other choice: DefaultExpander(p) with `p` being a Integer (number of included sectors) or a Float (percentage of the full two tensor update).
- `maxdims`: Maximal bond dimension
- `n_sweeps`: Number of full sweeps through the network. A full sweep contains a forward and backward sweep such that every tensor is optimized twice.
- `weight`: weights for the orthogonalization against `psi_ortho`.
- `eigsolve_tol`: Tolerance of the eigsolve procedure
- `eigsolve_krylovdim`: dimensionality of the krylov space
- `eigsolve_maxiter`: maximal iterations for the krylov algorithm
"""
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

function exponentiate_two_pass(H::Function, v, dt::Number, m::Int; 
                              tol::Real=1e-12, verbose::Bool=false)
    # Pass 1: Build tridiagonal matrix without storing basis
    beta0 = norm(v)
    beta0 < tol && return zero(v)
    
    # Initialize Lanczos vectors
    q_prev = zero(v)
    q_curr = v / beta0
    alphas = zeros(m)
    betas = zeros(m-1)
    # println(Base.summarysize(TTN.convert_cpu(v)))
    # CUDA.pool_status()
    
    # Lanczos iteration
    actual_m = m
    for j in 1:m
        println("j=", j)
        # println("q_prev:", Base.summarysize(TTN.convert_cpu(q_prev)))
        # println("q_curr:", Base.summarysize(TTN.convert_cpu(q_curr)))
        # println("result:", Base.summarysize(TTN.convert_cpu(result)))
        CUDA.poolJstatus()

        # Apply Hamiltonian
        
        w = H(q_curr)
        #  println(" =============== create w ===================== ")
        # println(Base.summarysize(TTN.convert_cpu(w)))
        # CUDA.pool_status()
        
        # Orthogonalize against previous vectors
        if j > 1
            w -= betas[j-1] * q_prev
        end
        
        # Compute alpha
        alphas[j] = real(dot(q_curr, w))  # Real for Hermitian H
        
        # Form new vector
        w -= alphas[j] * q_curr
        
        # Check for breakdown
        if j < m
            beta_j = norm(w)
            if beta_j < tol
                actual_m = j
                verbose && println("Early breakdown at m=$j")
                break
            end
            betas[j] = beta_j
            
            # Update vectors
            q_prev, q_curr = q_curr, w / beta_j
        end
    end
    
    # Adjust sizes if early breakdown
    alphas = alphas[1:actual_m]
    betas = betas[1:min(actual_m-1, length(betas))]
    
    # Build tridiagonal matrix
    T = actual_m == 1 ? [alphas[1]] : SymTridiagonal(alphas, betas)
    
    # Compute exponential action on first basis vector
    e1 = zeros(actual_m)
    e1[1] = 1.0
    u = exp(-im * dt * T) * e1
    
    # Pass 2: Reconstruct state
    result = zero(v)
    q_prev = zero(v)
    q_curr = v / beta0
    prev_beta = beta0
    
    # Accumulate result
    for j in 1:actual_m
        # println("j=", j)
        # println("q_prev:", Base.summarysize(TTN.convert_cpu(q_prev)))
        # println("q_curr:", Base.summarysize(TTN.convert_cpu(q_curr)))
        # println("result:", Base.summarysize(TTN.convert_cpu(result)))
        # Add current vector contribution
        # println(Base.summarysize(TTN.convert_cpu(result)))
        # CUDA.pool_status()
        result += u[j] * q_curr
        
        # Stop if last vector
        j == actual_m && break
        
        # Compute next vector
        w = H(q_curr)
        w -= alphas[j] * q_curr
        w -= prev_beta * q_prev  # Safe for j=1 (q_prev=0)
        println("w:", Base.summarysize(TTN.convert_cpu(q_prev)))
        
        # Normalize and update
        q_next = w / betas[j]
        q_prev, q_curr = q_curr, q_next
        prev_beta = betas[j]
    end
    
    # Scale by initial norm
    return beta0 * result, nothing
end

"""
```julia
   tdvp(psi0::TreeTensorNetwork, tpo::AbstractTensorProductOperator; kwargs...)
```

Performs a tdvp simulation of the state `psi0` with respect to the local Hamiltonian `tpo` defined either as a MPOWrapper or TPO object.
The integration is based on \$-i\\partial_t\\psi  = H\\psi\$, and thus describes a real time evolution.

It returns a sweep object `sp` where one can extract the final tree with `sp.ttn`.

# Keywords:
- `timestep`: time step for the integrator
- `initialtime`: starting time for the integrator
- `finaltime`: final time for the integrator
- `eigsolve_tol`: Tolerance of the eigsolve procedure
- `eigsolve_krylovdim`: dimensionality of the krylov space
- `eigsolve_maxiter`: maximal iterations for the krylov algorithm
"""
function tdvp(psi0::TreeTensorNetwork, tpo::AbstractTensorProductOperator; kwargs...)
    eigsolve_tol = get(kwargs, :eigsovle_tol, DEFAULT_TOL_TDVP)
    eigsolve_krylovdim = get(kwargs, :eigsolve_krylovdim, DEFAULT_KRYLOVDIM_TDVP)
    eigsolve_maxiter = get(kwargs, :eigsolve_maxiter, DEFAULT_MAXITER_TDVP)
    #eigsolve_verbosity = get(kwargs, :eigsolve_verbosity, DEFAULT_VERBOSITY_TDVP)
    ishermitian = get(kwargs, :ishermitian, DEFAULT_ISHERMITIAN_TDVP)
    eigsolve_eager = get(kwargs, :eager, DEFAULT_EAGER_TDVP)

    timestep = get(kwargs, :timestep, 1e-2)
    initialtime = get(kwargs, :initialtime, 0.)
    finaltime = get(kwargs, :finaltime, 1.)

    save_to_cpu = get(kwargs, :save_to_cpu, false)

    psic = TTN.gpu(copy(psi0))
    psic = move_ortho!(psic, (number_of_layers(network(psic)),1))

    pTPO = ProjectedTensorProductOperator(psic, tpo; save_to_cpu)

    if !save_to_cpu
        # func = (action, dt, T) -> exponentiate(action, convert(eltype(T), -1im*dt), T,
        #                                        krylovdim = eigsolve_krylovdim,
        #                                        tol = eigsolve_tol, 
        #                                        maxiter = eigsolve_maxiter,
        #                                        ishermitian = ishermitian,
        #                                        eager = eigsolve_eager);  
        func = (action, dt, T) -> exponentiate_twopass(action, convert(eltype(T), -1im*dt), T,
                                               krylovdim = eigsolve_krylovdim,
                                               # tol = eigsolve_tol, 
                                               # maxiter = eigsolve_maxiter,
                                               # ishermitian = ishermitian,
                                               # eager = false,
                                               );  
    else
        psic = TTN.cpu(psic)
        # func = (action, dt, T) -> exponentiate_twopass(action, convert(eltype(T), -1im*dt), T,
        #                                        krylovdim = eigsolve_krylovdim,
        #                                        # tol = eigsolve_tol, 
        #                                        # maxiter = eigsolve_maxiter,
        #                                        # ishermitian = ishermitian,
        #                                        # eager = false,
        #                                        );  

        function func(action, dt, T) 
            res = exponentiate_twopass(action, convert(eltype(T), -1im*dt), T,
                                               krylovdim = eigsolve_krylovdim,
                                               # tol = eigsolve_tol, 
                                               # maxiter = eigsolve_maxiter,
                                               # ishermitian = ishermitian,
                                               # eager = false,
                                               );  
            ref = exponentiate(action, convert(eltype(T), -1im*dt), T,
                                               krylovdim = eigsolve_krylovdim,
                                               tol = eigsolve_tol, 
                                               maxiter = eigsolve_maxiter,
                                               ishermitian = ishermitian,
                                               eager = eigsolve_eager);  
            norm_diff = norm(array(first(res)) .- array(first(ref))) / norm(array(first(ref)))
            if norm_diff>1e-10
                println("norm diff: ", norm_diff)
            end
            return res
        end
        
    end

    return sweep(psic, TDVPSweepHandler(psic, pTPO, timestep, initialtime, finaltime, func; save_to_cpu=save_to_cpu); kwargs...);
end
