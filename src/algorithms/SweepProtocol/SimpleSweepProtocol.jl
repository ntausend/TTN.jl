mutable struct SimpleSweepProtocol <: AbstractRegularSweepProtocol
    const net::AbstractNetwork
    const number_of_sweeps::Int
        
    dir::Symbol
    current_sweep::Int
    SimpleSweepProtocol(net, n_sweeps) = new(net, n_sweeps, :up, 1)
end


# simple reset the sweep protocol and update the current sweep number
# current number still needed?
function update!(sp::SimpleSweepProtocol)
    sp.dir = :up
    sp.current_sweep += 1 
    return sp
end


function next_position(sp::SimpleSweepProtocol, cur_pos::Tuple{Int,Int})
    cur_layer, cur_p = cur_pos
    if sp.dir == :up
        max_pos = number_of_tensors(sp.net, cur_layer)
        cur_p < max_pos && return (cur_layer, cur_p + 1)
        if cur_layer == number_of_layers(sp.net)
            sp.dir = :down
            return (cur_layer - 1, number_of_tensors(sp.net, cur_layer - 1))
        end
        return (cur_layer + 1, 1)
    elseif sp.dir == :down
        cur_p > 1 && return (cur_layer, cur_p - 1)
        cur_layer == 1 && return nothing
        return (cur_layer - 1, number_of_tensors(sp.net, cur_layer - 1))
    end
    error("Invalid direction of the iterator: $(sp.dir)")
end