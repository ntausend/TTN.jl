function expintegrator_twopass(A, t::Number, u::Tuple, alg::Lanczos)
    # Only Hermitian case, fixed finite t
    length(u) == 1 && return expintegrator_twopass(A, t, (u[1], zerovector(u[1])), alg)

    p = length(u) - 1 # = 0
    u₀ = first(u)
    β₀ = norm(u₀)
    β₀ < alg.tol && return zero(u₀), ConvergenceInfo(1, nothing, β₀, 0, 0)

    # Determine types
    Au₀ = apply(A, u₀)
    numops = 1
    T = promote_type(typeof(t), (typeof.(inner.(u, (Au₀,))))...)
    w₀ = scale(u₀, one(T))
    w = Vector{typeof(w₀)}(undef, p + 1)
    w[1] = w₀
    w[2] = scale!!(zerovector(w₀), Au₀, one(T))

    # Build polynomial combination w[p+1] = A^p u + ...
    for j in 1:p
        if j > 1
            w[j+1] = apply(A, w[j])
            numops += 1
        end
        lfac = 1
        for l in 0:(p-j)
            w[j+1] = add!!(w[j+1], u[j+l+1], (t)^l / lfac)
            lfac *= l + 1
        end
    end

    β = norm(w[p+1])
    β < alg.tol && return w₀, ConvergenceInfo(1, nothing, β, 0, numops)
    q_curr = scale(w[p+1], 1/β)
    q_prev = zero(q_curr)

    # Allocate tridiagonal coefficients
    m = alg.krylovdim
    alphas = zeros(real(T), m)
    betas  = zeros(real(T), m - 1)
    norms = zeros(real(T), m + 1)
    norms[1] = β₀
    norms[2] = β
    actual_m = m

    # Pass 1: Build tridiagonal matrix
    for j in 1:m
        wj = apply(A, q_curr)
        numops += 1
        α = real(dot(q_curr, wj))
        alphas[j] = α
        wj .-= α * q_curr
        if j > 1
            wj .-= betas[j-1] * q_prev
        end
        wj .-= dot(q_prev, wj) * q_prev
        wj .-= dot(q_curr, wj) * q_curr

        βj = norm(wj)
        if βj < alg.tol
            actual_m = j
            break
        end
        if j < m
            betas[j] = βj
            norms[j+2] = βj
            q_prev, q_curr = q_curr, wj / βj
        end
    end

    alphas = alphas[1:actual_m]
    betas  = betas[1:max(0, actual_m-1)]
    Tmat = actual_m == 1 ? [alphas[1]] : SymTridiagonal(alphas, betas)

    # Compute exp(t T) e₁
    e1 = zeros(complex(T), actual_m)
    e1[1] = 1.0
    uvec = exp(t * Tmat) * e1

    # Pass 2: Reconstruct result = ∑ uvec[j] * q_j
    result = zero(w₀)
    q_curr = scale(w[p+1], 1 / norms[2])
    q_prev .= zero(q_curr)

    for j in 1:actual_m
        result .+= uvec[j] * q_curr
        j == actual_m && break
        wj = apply(A, q_curr)
        numops += 1
        wj .-= alphas[j] * q_curr
        if j > 1
            wj .-= betas[j-1] * q_prev
        end
        wj .-= dot(q_prev, wj) * q_prev
        wj .-= dot(q_curr, wj) * q_curr

        β_actual = norm(wj)
        β_expected = j <= length(betas) ? betas[j] : 0.0
        if abs(β_actual - β_expected) > 1e-4
            break
        end
        q_prev, q_curr = q_curr, wj / β_expected
    end

    return β₀ * result, ConvergenceInfo(1, nothing, 0.0, 2*actual_m, numops)
end


using LinearAlgebra

"""
    expintegrator_twopass(A, t::Number, v, ::Lanczos;
                          krylovdim::Int=30, tol::Real=1e-12,
                          verbosity::Int=SILENT_LEVEL)

Compute y ≈ exp(t*A) * v for Hermitian A using a memory‑efficient 2‑pass Lanczos scheme.
Assumptions: Hermitian A, finite fixed t, p = 0 (no inhomogeneous terms), no adaptivity, no restarts.

Returns: y, ConvergenceInfo(converged, nothing, normres, numiter, numops)
- converged = 1
- normres   = 0.0 (no explicit error estimate here)
- numiter   = 1   (single Krylov build)
- numops    = # of applications of A
"""
function expintegrator_twopass(A, t::Number, v, ::Lanczos;
                               krylovdim::Int=30, tol::Real=1e-12,
                               verbosity::Int=SILENT_LEVEL)

    # Early exit: zero input
    β0 = norm(v)
    β0 < tol && return zero(v), ConvergenceInfo(1, nothing, β0, 0, 0)

    # Types and basic buffers
    Tsc = promote_type(typeof(t), eltype(v))
    q_prev = zero(v)
    q_curr = (1/β0) * v
    m = krylovdim

    # Tridiagonal coefficients (real for Hermitian A)
    α = zeros(real(Tsc), m)
    β = zeros(real(Tsc), max(m-1, 0))

    numops = 0
    actual_m = m

    # ─────────────
    # Pass 1: Lanczos (build α, β) with light reorthogonalization
    # ─────────────
    for j in 1:m
        w = A(q_curr); numops += 1

        aj = real(dot(q_curr, w))
        α[j] = aj
        w .-= aj * q_curr
        if j > 1
            w .-= β[j-1] * q_prev
        end
        # optional light reorthogonalization
        if j > 1
            w .-= dot(q_prev, w) * q_prev
        end
        w .-= dot(q_curr, w) * q_curr

        bj = norm(w)
        if bj < tol
            actual_m = j
            break
        end
        if j < m
            β[j] = bj
            q_prev, q_curr = q_curr, (1/bj) .* w
        end
    end

    α = α[1:actual_m]
    β = β[1:max(actual_m-1, 0)]
    Ttri = actual_m == 1 ? [α[1]] : SymTridiagonal(α, β)

    # Small exponential: u = exp(t*T) * e1
    e1 = zeros(complex(Tsc), actual_m); e1[1] = 1
    u = exp(t * Ttri) * e1

    # ─────────────
    # Pass 2: Reconstruct y = β0 * Σ u[j] q_j via recurrence (no storing of full basis)
    # ─────────────
    y = zero(v)
    q_prev .= zero(v)
    q_curr .= (1/β0) .* v
    y .+= u[1] .* q_curr

    for j in 2:actual_m
        w = A(q_curr); numops += 1
        w .-= α[j-1] * q_curr
        w .-= β[j-2] * q_prev
        # same light reorthogonalization
        w .-= dot(q_prev, w) * q_prev
        w .-= dot(q_curr, w) * q_curr

        bj = norm(w)
        # If numerical drift makes bj ~ 0, stop gracefully
        bj < tol && break

        q_next = (1/bj) .* w
        y .+= u[j] .* q_next
        q_prev, q_curr = q_curr, q_next
    end

    y .*= β0
    return y, ConvergenceInfo(1, nothing, 0.0, 1, numops)
end

# Convenience wrapper to match KrylovKit's exponentiate signature:
exponentiate(A, t::Number, v, alg::Lanczos; kwargs...) =
    expintegrator_twopass(A, t, v, alg; kwargs...)
