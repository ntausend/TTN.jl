function dmrg(psi0::TreeTensorNetwork, mpo::AbstractTensorProductOperator; kwargs...)
    n_sweeps = get(kwargs, :number_of_sweeps, 1)


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

    for p in sp
        @assert p == ortho_center(psic)
        # optimize current position
        t = psic[p]
        action = ∂A(pTPO, p)
        val, tn = eigsolve(action, t, 1, :SR)
        push!(eivals, real(val[1]))
        tn = tn[1]

        # get next step
        pn = next_position(sp,p)

        if(!isnothing(pn))
            #save the tensor
            psic[p] = tn
            # orthogonalize to the next position
            move_ortho!(psic, pn)
            # correct the environment
            pth = connecting_path(net, p, pn)
            pth = vcat(p, pth)
            for (jj,pk) in enumerate(pth[1:end-1])
                ism = psic[pk]
                pTPO = update_environments!(pTPO, ism, pk, pth[jj+1])
            end
        else
            break
        end
    end

    return eivals, psic
end


abstract type AbstractSweepProtocol end


start_position(::AbstractSweepProtocol)  = (1,1)
next_position(::AbstractSweepProtocol, ::Tuple{Int, Int}) = nothing

function Base.iterate(sp::AbstractSweepProtocol)
    pos = start_position(sp)
    return (pos, pos)
end

function Base.iterate(sp::AbstractSweepProtocol, state)
    next_pos = next_position(sp, state)
    isnothing(next_pos) && return nothing
    return (next_pos, next_pos)
end


struct SimpleSweepProtocol <: AbstractSweepProtocol
    net::AbstractNetwork
    number_of_sweeps::Int
    n_sweeps::Vector{Int}
    dir::Vector{Int}
    SimpleSweepProtocol(net, n_sweeps) = new(net, n_sweeps, [0], [1])
end

function next_position(sp::SimpleSweepProtocol, cur_pos::Tuple{Int,Int})
    cur_layer = cur_pos[1]
    cur_pos   = cur_pos[2]


    if sp.dir[1] == 1
        max_pos   = number_of_tensors(sp.net, cur_layer)
        cur_pos < max_pos && return (cur_layer, cur_pos + 1)
        if cur_layer == number_of_layers(sp.net)
            sp.dir[1] = -1
            return (cur_layer - 1, number_of_tensors(sp.net, cur_layer - 1))
        end
        return (cur_layer + 1, 1)
    else
        cur_pos > 1 && return (cur_layer, cur_pos - 1)
        if cur_layer == 1
            sp.dir[1] = 1
            sp.n_sweeps[1] += 1 
            sp.n_sweeps[1] == sp.number_of_sweeps && return nothing
            return (cur_layer, 2)
        end
        return (cur_layer - 1, number_of_tensors(sp.net, cur_layer - 1))
    end
end