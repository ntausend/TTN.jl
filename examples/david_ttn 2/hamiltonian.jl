function hamiltonian_tfi(; N::Int64, h::Int64, pbc::Bool = false)
    ampo = OpSum()

    for jj in 1:N-1
        ampo += (h, "Z", jj)
        ampo += (1, "X", jj, "X", jj+1)
    end
    ampo += (h, "Z", N)

    if pbc
        ampo += (1, "X", N, "X", 1)
    end

    return ampo
end
