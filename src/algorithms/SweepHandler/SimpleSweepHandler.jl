mutable struct SimpleSweepHandler <: AbstractRegularSweepHandler
    const number_of_sweeps::Int
    ttn::TreeTensorNetwork
    pTPO::ProjTensorProductOperator
    func
        
    dir::Symbol
    current_sweep::Int
    energies::Vector{Float64}
    SimpleSweepHandler(ttn, pTPO, func, n_sweeps) = new(n_sweeps, ttn, pTPO, func, :up, 1, Float64[])
end

function initialize!(sp::SimpleSweepHandler)
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

# simple reset the sweep Handler and update the current sweep number
# current number still needed?
function update_next_sweep!(sp::SimpleSweepHandler)
    sp.dir = :up
    sp.current_sweep += 1 
    return sp
end

function update!(sp::SimpleSweepHandler, pos::Tuple{Int, Int})
    @assert pos == ortho_center(sp.ttn)
    ttn = sp.ttn
    pTPO = sp.pTPO
    
    net = network(ttn)

    t = ttn[pos]
    action = ∂A(pTPO, pos)
    val, tn = sp.func(action, t)
    push!(sp.energies, real(val[1]))

    #save the tensor
    ttn[pos] = tn[1]

    pn = next_position(sp,pos)
    net = network(sp.ttn)

    if !isnothing(pn)
        move_ortho!(ttn, pn)
        pth = connecting_path(net, pos, pn)
        pth = vcat(pos, pth)
        for (jj,pk) in enumerate(pth[1:end-1])
            ism = ttn[pk]
            pTPO = update_environments!(pTPO, ism, pk, pth[jj+1])
        end
    end
end


function next_position(sp::SimpleSweepHandler, cur_pos::Tuple{Int,Int})
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