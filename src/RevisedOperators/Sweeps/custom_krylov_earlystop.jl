using LinearAlgebra
using SparseArrays

# exponentiate_twopass(A, t::Number, v; kwargs...) = expintegrator_twopass(A, t, v; kwargs...)
function exponentiate_twopass(H, t::Number, v; krylovdim=30, tol=1e-12)
    m = krylovdim
    # Pass 1: Build tridiagonal matrix
    beta0 = norm(v)
    beta0 < tol && return zero(v), ConvergenceInfo(1, nothing, beta0, 0, 0)

    # Initialize Lanczos vectors
    q_prev = v
    q_curr = v / beta0
    alphas = zeros(m)
    betas  = zeros(m-1)
    numops = 0

    # For accumulated total error (single fixed step ⇒ equals final accepted ε)
    totalerr = 0.0
    last_eps = 0.0

    # Lanczos iteration
    actual_m = m
    for j in 1:m
        w = H(q_curr)
        numops += 1

        # Orthogonalization
        alpha = real(dot(q_curr, w))  # Ensure real for Hermitian
        alphas[j] = alpha
        w -= alpha * q_curr

        if j > 1
            w -= betas[j-1] * q_prev
        end

        # Reorthogonalization (improves stability)
        if j > 1
            w -= dot(q_prev, w) * q_prev
        end
        w -= dot(q_curr, w) * q_curr

        # Compute next beta
        beta = norm(w)

        # --- Early stopping test (Krylov residual–based error estimate) ---
        # Build tiny T_j from {α₁..αⱼ, β₁..βⱼ₋₁}
        Tj = (j == 1) ? Diagonal(@view alphas[1:1]) : SymTridiagonal(@view(alphas[1:j]), @view(betas[1:j-1]))
        # Build augmented (j+2)×(j+2) matrix:
        # H_aug = [ t*Tj  e₁  0;  0  0  1;  0  0  0 ]
        Tj_mat = Matrix(t .* Tj)
        Haug = zeros(eltype(Tj_mat), j + 2, j + 2)
        Haug[1:j, 1:j] .= Tj_mat
        Haug[1, j + 1] = one(eltype(Tj_mat))
        Haug[j + 1, j + 2] = one(eltype(Tj_mat))
        expHaug = exp(Haug)
        # ε_j = |β₀| * |β_j| * |exp(H_aug)[j, j+2]|
        eps_j = abs(beta0) * abs(beta) * abs(expHaug[j, j + 2])
        last_eps = eps_j

        # Accept early if ε_j ≤ |t| * tol
        if eps_j <= abs(t) * tol
            totalerr += eps_j
            actual_m = j
            break
        end
        # ---------------------------------------------------------------

        if beta < tol
            totalerr += eps_j
            actual_m = j
            break
        end

        if j < m
            betas[j] = beta
            # q_next = w / beta
            q_prev, q_curr = q_curr, w / beta
        end
        actual_m = j
    end

    # If we used full m without early acceptance/breakdown, use last_eps
    if totalerr == 0.0
        totalerr += last_eps
    end

    # Adjust sizes if early termination
    alphas = alphas[1:actual_m]
    betas = betas[1:min(actual_m-1, length(betas))]

    # Build tridiagonal matrix
    T = actual_m == 1 ? Diagonal([alphas[1]]) : SymTridiagonal(alphas, betas)

    # Compute exponential action on first basis vector
    e1 = zeros(complex(eltype(v)), actual_m)
    e1[1] = 1.0
    u = exp(t * T) * e1

    # Pass 2: Reconstruct solution
    result = zero(v)
    q_prev = v
    q_curr = v / beta0
    # broke_consistency = false

    # Accumulate result
    for j in 1:actual_m
        # Add current vector contribution
        result .+= u[j] .* q_curr
        # Stop after last vector
        j == actual_m && break

        # Compute next vector
        w = H(q_curr)
        numops += 1

        w .-= alphas[j] .* q_curr
        if j > 1
            w .-= betas[j-1] .* q_prev
        end

        # Reorthogonalize in second pass
        if j > 1
            w .-= dot(q_prev, w) .* q_prev
        end
        w .-= dot(q_curr, w) .* q_curr

        beta_expected = (j <= length(betas)) ? betas[j] : 0
        beta_expected == 0 && break # exact break from pass 1

        # loss of orthogonality doesn't matter for exponential as long as error is smaller
        # otherwise catch by
        # beta_actual = norm(w)
        # if abs(beta_actual - beta_expected) > 1e-4
        #     broke_consistency = true
        #     break
        # end

        # Prepare next iteration
        # q_next = w / betas[j]
        q_prev, q_curr = q_curr, w / betas[j]
    end

    converged = (totalerr <= abs(t) * tol) ? 1 : 0
    # converged = (!broke_consistency && totalerr <= abs(t) * tol) ? 1 : 0
    return beta0 .* result, ConvergenceInfo(converged, nothing, totalerr, 1, numops)
end

"""
    eigsolve_twopass(H, x0;
        howmany=1, which=:SR, krylovdim=30, tol=1e-12,
        reorth=:local, beta_check_tol=1e-6)

Memory-efficient Lanczos eigensolver for Hermitian linear map `H(::T)->T`, with two passes:
  - Pass 1: build SymTridiagonal T (size ≤ krylovdim) via 3-term Lanczos; store only α, β.
  - Pass 2: reconstruct eigenvectors x ≈ Q_m y using the recorded Ritz vectors y of T.

Returns: vals::Vector, vecs::Vector{typeof(x0)}, info::NamedTuple

Stable only for howmany = 1
For more eigenpairs might give multiple copies of the same eigenpair due to loss of orthogonality
Implement stronger reorthogonalization and restarts for more stable version
"""
function eigsolve_twopass(H, x0;
    howmany::Int=1, which::Symbol=:SR, krylovdim::Int=30, tol::Real=1e-12,
    reorth::Symbol=:local, beta_check_tol::Real=1e-10)

    apply_H(v) = isa(H, AbstractMatrix) ? H * v : H(v)

    # ---- Pass 1: Lanczos to build T = tridiag(betas, alphas, betas) ----
    m = krylovdim
    β0 = norm(x0)
    β0 < tol && return Float64[], typeof(x0)[], ConvergenceInfo(1, nothing, 0.0, 0, 0)

    q_prev = zero(x0)
    q_curr = x0 / β0

    alphas = zeros(Float64, m)
    betas  = zeros(Float64, max(m-1,0))

    numops = 0
    actual_m = 0
    β_tail = 0.0  # β_m (needed for residuals of Ritz pairs)

    for j in 1:m
        w = apply_H(q_curr); numops += 1

        α = real(dot(q_curr, w))
        alphas[j] = α
        w .-= α .* q_curr
        if j > 1
            w .-= betas[j-1] .* q_prev
        end

        # optional local reorth for stability
        if reorth == :local
            if j > 1
                w .-= dot(q_prev, w) .* q_prev
            end
            w .-= dot(q_curr, w) .* q_curr
        end

        β = norm(w)
        β_tail = β
        actual_m = j

        early_stop = false
        if β <= tol
            early_stop = true
        elseif j >= howmany
            # Build the current T_j and test residuals r_i = |β * e_j' * y_i|
            Tj = (j == 1) ? SymTridiagonal([alphas[1]], Float64[]) :
                            SymTridiagonal(view(alphas, 1:j), view(betas, 1:j-1))
            Fj = eigen(Matrix(Tj))             # small j×j eigenproblem
            # choose ordering
            order = if which == :SR
                collect(1:length(Fj.values))
            elseif which == :LR
                collect(length(Fj.values):-1:1)
            elseif which == :LM
                sortperm(abs.(Fj.values); rev=true)
            else
                error("which = $which not supported in this Hermitian two-pass variant")
            end
            pickj = order[1:howmany]
            Yj    = Fj.vectors[:, pickj]       # j × howmany
            rj    = abs.(β .* view(Yj, j, :))  # residual proxies
            convj = count(<=(tol), rj)
            if convj >= howmany
                early_stop = true
            end
        end

        if early_stop
            # we intentionally do NOT form q_next; actual_m=j and β_tail=β are set
            break
        end
        # -------------------------------

        if j == m
            break
        else
            betas[j] = β
            q_next = w / β
            q_prev, q_curr = q_curr, q_next
        end
    end

    # Trim sizes
    alphas = alphas[1:actual_m]
    betas  = betas[1:max(actual_m-1,0)]

    # Build T and get Ritz pairs
    T = (actual_m == 1) ? SymTridiagonal([alphas[1]], Float64[]) :
                          SymTridiagonal(alphas, betas)
    F = eigen(Matrix(T))           # eigenvalues ascending; columns are Ritz vectors y
    evals_all = F.values
    Y = F.vectors                  # size actual_m × actual_m

    # Select which eigenvalues to return
    function sel_order(vals, which::Symbol)
        if which == :SR      # smallest real part (Hermitian ⇒ smallest)
            return collect(1:length(vals))
        elseif which == :LR  # largest real part
            return collect(length(vals):-1:1)
        elseif which == :LM  # largest magnitude
            return sortperm(abs.(vals); rev=true)
        else
            error("which = $which not supported in this Hermitian two-pass variant")
        end
    end
    order = sel_order(evals_all, which)
    pick = order[1:howmany]
    vals = evals_all[pick]
    Ysel = Y[:, pick]             # columns = Ritz vectors y^{(i)}, i=1..howmany

    # Residual norms: ||H x_i - λ_i x_i|| = |β_m * e_m^T y^{(i)}|
    emY = Ysel[end, :]            # last row of selected Ritz vectors
    normres = abs.(β_tail .* emY)

    # Convergence count
    converged = count(<=(tol), normres)

    # ---- Pass 2: reconstruct eigenvectors x_i = Q_m y_i without storing Q_m ----
    vecs = [zero(x0) for _ in 1:howmany]

    q_prev = zero(x0)
    q_curr = x0 / β0
    # stream through basis and accumulate x_i += y_i[j] * q_j
    for j in 1:actual_m
        # accumulate current basis vector contribution
        @inbounds for i in 1:howmany
            vecs[i] .+= (Ysel[j, i]) .* q_curr
        end

        # last basis vector done
        j == actual_m && break

        # move recurrence forward (use recorded α_j, β_j)
        w = apply_H(q_curr); numops += 1
        w .-= alphas[j] .* q_curr
        if j > 1
            w .-= betas[j-1] .* q_prev
        end

        # optional local reorth
        if reorth == :local
            if j > 1
                w .-= dot(q_prev, w) .* q_prev
            end
            w .-= dot(q_curr, w) .* q_curr
        end

        β_actual = norm(w)
        β_expected = betas[j]      # j ≤ actual_m-1 here

        # keep recurrence numerically consistent with pass-1
        if β_expected > 0
            if abs(β_actual - β_expected) > max(beta_check_tol, 100*sqrt(eps(Float64))*β_expected)
                w .*= (β_expected / max(β_actual, eps(Float64)))
            end
            q_next = w / β_expected
        else
            break
        end

        q_prev, q_curr = q_curr, q_next
    end

    # Scale back because q1 = x0 / β0
    for i in 1:howmany
        vecs[i] .*= β0
    end

    # build residual vectors and norms, update convergence
    # residuals = Vector{typeof(x0)}(undef, howmany)
    # normres   = zeros(Float64, howmany)
    # for i in 1:howmany
    #     r = apply_H(vecs[i]);  numops += 1
    #     @. r = r - vals[i] * vecs[i]
    #     residuals[i] = r
    #     normres[i]   = norm(r)
    # end
    # converged = count(<=(tol), normres)

    # # replace your previous `info = ConvergenceInfo(converged, nothing, collect(normres), 1, numops)`
    # info = ConvergenceInfo(converged, residuals, normres, 1, numops)

    # return vals, vecs, info

    info = ConvergenceInfo(converged, nothing, collect(normres), 1, numops)

    return vals, vecs, info
end
