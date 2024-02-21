function main(n_sweeps::Int64, maxdim::Int64; 
        mdl_params, maxiter_expander = 2, noise = 0, p = 0.1, conserve_qns = false, cutoff = 1E-13, use_random_init = true)

        nsites = mdl_params[:N]

        # this defines the local hilbertspace, independent on the state ansatz.
        s = siteinds("SpinHalf", nsites; conserve_szparity = conserve_qns)

        # this defines the network to use the state ansatz
        net = TTNKit.TrenaryChainNetwork(s)


        if use_random_init
                #ψ₀ = TTNKit.ProductTreeTensorNetwork(net, start_vector)
                if conserve_qns
                        # here you define the target charge for the random tree state 
                        target_charge = Qn("SzParity",0,2)
                        ψ₀ = TTNKit.RandomTreeTensorNetwork(net, target_charge; maxdim)
                else
                        ψ₀ = TTNKit.RandomTreeTensorNetwork(net; maxdim)
                end
        else
                # here you define the starting qn numbers. If unsymmetric minimization, it is basically irrelevant
                # however, choosing a good initialization could lead to improved convergence.
                # We also have a patron wavefunction initialization, but this is more evolved. For the simple
                # TFI this should be sufficient
                start_vector = fill("Dn", nsites)
                ψ₀ = TTNKit.ProductTreeTensorNetwork(net, start_vector)
        end


        H = Hamiltonian(hamiltonian_tfi(;model_params...), physical_lattice(net))

        tol_expander = 1E-5

        dim_krylov = 3
        maxiter_krylov = 1

        expander = TTNKit.DefaultExpander(p; maxiter = maxiter_expander, tol = tol_expander)
        sp = TTNKit.dmrg(ψ₀, H; expander = expander,
                     number_of_sweeps = n_sweeps, maxdims = maxdim, noise = noise, outputlevel = 3)

        ψ = sp.ttn


        return ψ
end
