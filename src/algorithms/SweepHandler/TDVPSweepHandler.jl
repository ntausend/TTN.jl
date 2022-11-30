mutable struct TDVPSweepHandler <: AbstractRegularSweepHandler
    const net::AbstractNetwork
    const finaltime::Float64
    const timestep::Float64
    ttn::TreeTensorNetwork
    pTPO::ProjTensorProductOperator

    dir::Symbol
    dir2::Symbol
    current_timestep::Int

    function TDVPSweepHandler(ttn, pTPO, timestep, finaltime)
        return new(finaltime, timestep, ttn, pTPO, :forward, :up, 0)
    end
end

sweeps(sp::TDVPSweepHandler) = 0:sp.timestep:sp.finaltime

function initialize!(sp::TDVPSweepHandler)
    ttn = sp.ttn
    pTPO = sp.pTPO

    net = network(ttn)

    # now move everything to the starting point
    ttn = move_ortho!(ttn, (1,1))

    # update the environments accordingly

    pth = connecting_path(net, (number_of_layers(net),1), (1,1))
    pth = vcat((number_of_layers(net),1), pth)
    for (jj,p) in enumerate(pth[1:end-1])
        ism = ttn[p]
        pTPO = update_environments!(pTPO, ism, p, pth[jj+1])
    end
end


function update_next_sweep!(sp::TDVPSweepHandler)
    return nothing
end

function update!(sp::TDVPSweepHandler, pos::Tuple{Int, Int})
    ttn = sp.ttn
    pTPO = sp.pTPO
    if sp.dir2 == :up
        T = ttn[pos]
        # time evolve tensor at pos_initial
        T = TensorKit.permute(T, Tuple(1:number_of_indices(T)), ())
        #eff_Hamiltonian = environment(pTPO, pos_initial)
        action = ∂A(pTPO, pos)
        Tn = TensorKit.permute(exponentiate(action, dt, T), 
                                Tuple(1:number_of_indices(T)-1), (number_of_indices(T),)
                            )

        # QR-decomposition of time evolved tensor at pos_initial
        new_Q_initial, old_R = leftorth(Tn)
        
        # reverse time evolution for R tensor between pos_initial and pos_final
        old_R = TensorKit.permute(old_R, (1,2))
        identity = TensorKit.id((domain(new_Q_initial))')

        # this will break
        #############################################
        eff_Hamiltonian = (adjoint(new_Q_initial) ⊗ identity)*eff_Hamiltonian*(new_Q_initial ⊗ identity)
        new_R = TensorKit.permute(exponentiate(eff_Hamiltonian, -dt, old_R), (1,), (2,))
        #############################################

        # multiply new R tensor on tensor at pos_final
        idx = index_of_child(net, pos_initial)
        idx_dom, idx_codom = split_index(net, pos_final, idx)
        perm = vcat(idx_dom..., idx_codom...)
        new_Q_final = TensorKit.permute(new_R*TensorKit.permute(ttn[pos_final], idx_dom, idx_codom), Tuple(perm[1:end-1]), (perm[end],))

        # set new tensors
        ttn[pos_initial] = new_Q_initial
        ttn[pos_final]   = new_Q_final

        # move orthocenter (just for consistency), update ttnc and environments  
        ttn.ortho_center[1] = pos_final[1]
        ttn.ortho_center[2] = pos_final[2]

        update_environment!(pTPO, new_Q_initial, pos, pos_final)

        # check if dir2 get reversed

    else

    end
end
function next_position(sp::TDVPSweepHandler, cur_pos::Tuple{Int,Int})
    # this you may need to change
    cur_layer, cur_p = cur_pos
    net = network(sp.ttn)
    if sp.dir == :up
        max_pos = number_of_tensors(net, cur_layer)
        cur_p < max_pos && return (cur_layer, cur_p + 1)
        if cur_layer == number_of_layers(net)
            sp.dir = :down
            return (cur_layer - 1, number_of_tensors(net, cur_layer - 1))
        end
        return (cur_layer + 1, 1)
    elseif sp.dir == :down
        cur_p > 1 && return (cur_layer, cur_p - 1)
        cur_layer == 1 && return nothing
        return (cur_layer - 1, number_of_tensors(net, cur_layer - 1))
    end
    error("Invalid direction of the iterator: $(sp.dir)")
end

