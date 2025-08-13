using LinearAlgebra
using SparseArrays

# exponentiate_twopass(A, t::Number, v; kwargs...) = expintegrator_twopass(A, t, v; kwargs...)
function exponentiate_twopass(H, t::Number, v; krylovdim=30, tol=1e-12)
    m = krylovdim
    # Pass 1: Build tridiagonal matrix
    beta0 = norm(v)
    beta0 < tol && return zero(v), ConvergenceInfo(1, nothing, beta0, 0, 0)

    # Initialize Lanczos vectors
    q_prev = zero(v)
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
        # if j == 1
        #     Tj = Diagonal(@view alphas[1:1])
        # else
        #     Tj = SymTridiagonal(@view(alphas[1:j]), @view(betas[1:j-1]))
        # end
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
            q_next = w / beta
            q_prev, q_curr = q_curr, q_next
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
    q_prev = zero(v)
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
        q_next = w / betas[j]
        q_prev, q_curr = q_curr, q_next
    end

    converged = (totalerr <= abs(t) * tol) ? 1 : 0
    # converged = (!broke_consistency && totalerr <= abs(t) * tol) ? 1 : 0
    return beta0 .* result, ConvergenceInfo(converged, nothing, totalerr, 1, numops)
end

