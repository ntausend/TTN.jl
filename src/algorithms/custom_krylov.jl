using LinearAlgebra
using SparseArrays

# exponentiate_twopass(A, t::Number, v; kwargs...) = expintegrator_twopass(A, t, v; kwargs...)
function exponentiate_twopass(H, t::Number, v; krylovdim=30, tol=1e-5)
    m = krylovdim
    # Pass 1: Build tridiagonal matrix
    beta0 = norm(v)
    beta0 < tol && return zero(v)

    # Initialize Lanczos vectors
    q_prev = zero(v)
    q_curr = v / beta0
    alphas = zeros(m)
    betas  = zeros(m-1)
    norms = zeros(m+1)
    norms[1] = beta0

    # Lanczos iteration
    actual_m = m
    for j in 1:m
        w = H(q_curr)

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
        if beta < tol
            actual_m = j
            break
        end

        if j < m
            betas[j] = beta
            norms[j+1] = beta
            q_next = w / beta
            q_prev, q_curr = q_curr, q_next
        end
        actual_m = j
    end

    # Adjust sizes if early termination
    alphas = alphas[1:actual_m]
    betas = betas[1:min(actual_m-1, length(betas))]

    # Build tridiagonal matrix
    T = actual_m == 1 ? [alphas[1]] : SymTridiagonal(alphas, betas)

    # Compute exponential action on first basis vector
    e1 = zeros(complex(eltype(v)), actual_m)
    e1[1] = 1.0
    u = exp(t * T) * e1

    # Pass 2: Reconstruct solution
    result = zero(v)
    q_prev = zero(v)
    q_curr = v / norms[1]

    # Accumulate result
    for j in 1:actual_m
        # Add current vector contribution

        result .+= u[j] .* q_curr

        # Stop after last vector
        j == actual_m && break

        # Compute next vector
        w = H(q_curr)

        w .-= alphas[j] .* q_curr
        if j > 1
            w .-= betas[j-1] .* q_prev
        end

        # Reorthogonalize in second pass
        if j > 1
            w .-= dot(q_prev, w) .* q_prev
        end
        w .-= dot(q_curr, w) .* q_curr

        beta_actual = norm(w)
        beta_expected = j <= length(betas) ? betas[j] : 0

        if abs(beta_actual - beta_expected) > 1e-4
            break
        end

        # Prepare next iteration
        q_next = w / betas[j]
    end

    return beta0 .* result, nothing
end
