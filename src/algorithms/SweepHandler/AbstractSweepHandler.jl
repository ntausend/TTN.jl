abstract type AbstractSweepHandler end

initialize!(::AbstractSweepHandler) = nothing

start_position(::AbstractSweepHandler)  = (1,1)

current_sweep(::AbstractSweepHandler) = nothing
info_string(::AbstractSweepHandler, ::Int) = nothing

# returning the next position based on the protocol (and the state of the protocol)
# and the current position. Will be used in the iterator
next_position(::AbstractSweepHandler, ::Tuple{Int, Int}) = nothing

# updates the sweep handler to the next sweep
update_next_sweep!(sp::AbstractSweepHandler) = sp

# closure function for updating
function update!(::AbstractSweepHandler, pos::Tuple{Int, Int}, ttn::TreeTensorNetwork{N, T}, pTPO::AbstractProjTPO{N,T}, tn::T) where{N,T}
    ttn[pos] = tn
    nothing
end

# iterator over the number of sweeps. Keep it abstract and let the Handlers handle things
sweeps(::AbstractSweepHandler) = nothing 

function Base.iterate(sp::AbstractSweepHandler)
    pos = start_position(sp)
    return (pos, pos)
end
function Base.iterate(sp::AbstractSweepHandler, state)
    next_pos = next_position(sp, state)
    if isnothing(next_pos)
        update_next_sweep!(sp)
        return nothing
    end
    return (next_pos, next_pos)
end

# regular type with having a garanted field called number_of_sweeps
abstract type AbstractRegularSweepHandler <: AbstractSweepHandler end
number_of_sweeps(sp::AbstractRegularSweepHandler) = sp.number_of_sweeps
sweeps(sp::AbstractRegularSweepHandler) = 1:number_of_sweeps(sp)