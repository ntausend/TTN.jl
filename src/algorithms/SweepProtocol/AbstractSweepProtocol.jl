abstract type AbstractSweepProtocol end


start_position(::AbstractSweepProtocol)  = (1,1)

# returning the next position based on the protocol (and the state of the protocol)
# and the current position. Will be used in the iterator
next_position(::AbstractSweepProtocol, ::Tuple{Int, Int}) = nothing

# updates the sweep handler to the next sweep
update!(sp::AbstractSweepProtocol) = sp

# iterator over the number of sweeps. Keep it abstract and let the protocols handle thinks
sweeps(::AbstractSweepProtocol) = nothing 

function Base.iterate(sp::AbstractSweepProtocol)
    pos = start_position(sp)
    return (pos, pos)
end
function Base.iterate(sp::AbstractSweepProtocol, state)
    next_pos = next_position(sp, state)
    if isnothing(next_pos)
        update!(sp)
        return nothing
    end
    return (next_pos, next_pos)
end

# regular type with having a garanted field called number_of_sweeps
abstract type AbstractRegularSweepProtocol <: AbstractSweepProtocol end
number_of_sweeps(sp::AbstractRegularSweepProtocol) = sp.number_of_sweeps
sweeps(sp::AbstractRegularSweepProtocol) = 1:number_of_sweeps(sp)