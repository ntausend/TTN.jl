function dmrg(psi0::TreeTensorNetwork, mpo::AbstractTensorProductOperator; kwargs...)
    n_sweeps = get(kwargs, :number_of_sweeps, 1)
    verbose_level = get(kwargs, :verbose_level, 1)

    net = network(psi0)

    psic = copy(psi0)
    psic = move_ortho!(psic, (number_of_layers(net),1))

    pTPO = ProjTensorProductOperator(psi0, mpo)

    # now move everything to the starting point
    psic = move_ortho!(psic, (1,1))

    # update the environments accordingly

    pth = connecting_path(net, (number_of_layers(net),1), (1,1))
    pth = vcat((number_of_layers(net),1), pth)
    for (jj,p) in enumerate(pth[1:end-1])
        ism = psic[p]
        pTPO = update_environments!(pTPO, ism, p, pth[jj+1])
    end

    # now start with the sweeping protocol
    sp = SimpleSweepProtocol(net, n_sweeps)

    eivals = Float64[]

    for sw in sweeps(sp)
        if verbose_level ≥ 2 
            println("Start sweep number $(sw)")
            flush(stdout)
        end

        for pos in sp
            if verbose_level ≥ 3
                println("\tOptimizing position $(pos)")
                flush(stdout)
            end

            @assert pos == ortho_center(psic)
            t = psic[pos]
            action = ∂A(pTPO, pos)
            val, tn = eigsolve(action, t, 1, :SR)
            push!(eivals, real(val[1]))
            tn = tn[1]
            #save the tensor
            psic[pos] = tn
            # get next step
            pn = next_position(sp,pos)
            if verbose_level ≥ 3
                println("\tFinished optimizing position $(pos)")
                flush(stdout)
            end

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
        end
        if verbose_level ≥ 1
            println(repeat("=", 100))
            println("Finsihed sweep $(sw). Current energy: $(eivals[end])")
            println(repeat("=", 100))
            flush(stdout)
        end
    end
    return eivals, psic
end