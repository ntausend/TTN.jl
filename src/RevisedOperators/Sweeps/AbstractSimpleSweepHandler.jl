abstract type AbstractSimpleSweepHandler <: AbstractRegularSweepHandler end

current_sweep(sh::AbstractSimpleSweepHandler) = sh.current_sweep

function maxdim(sh::AbstractSimpleSweepHandler)
    length(sh.maxdims) < current_sweep(sh) && return sh.maxdims[end]
    return sh.maxdims[current_sweep(sh)]
end

function info_string(sh::AbstractSimpleSweepHandler, output_level::Int)
    e = sh.current_energy
    # trnc_wght = sh.current_max_truncerr
    # todo ->  make a function for that .... which also can handle TensorKit
    maxdim = maxlinkdim(sh.ttn)
    output_level ≥ 1 && @printf("\tCurrent energy: %.15f.\n", e)
    # output_level ≥ 2 && @printf("\tTruncated Weigth: %.3e. Maximal bond dim = %i\n", trnc_wght, maxdim)
    # sh.current_max_truncerr = 0.0
    nothing
end

# simple reset the sweep Handler and update the current sweep number
# current number still needed?
function update_next_sweep!(sp::AbstractSimpleSweepHandler)
    sp.dir = :up
    sp.current_sweep += 1 
    return sp
end

function next_position(sp::AbstractSimpleSweepHandler, cur_pos::Tuple{Int,Int})
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