struct HardCoreBosonNode{S<:IndexSpace,I<:Sector} <: PhysicalNode{S,I}
    s::Int
    hilbertspace::S
    desc::AbstractString
    function HardCoreBosonNode(pos::Int, desc::AbstractString="";
            conserve_qns::Bool = true, conserve_parity::Bool = conserve_qns)
        if conserve_qns
            sp = U1Space
            sectors = (0 => 1, 1 => 1)
        elseif conserve_parity
            sp = Z2Space
            sectors = (0 => 1, 1 => 1)
        else
            sp = ComplexSpace
            sectors = (2)
        end
        return new{sp, sectortype(sp)}(pos, sp(sectors), "HCB "*desc)
    end
end
#=
function _state(::Type{HardCoreBosonNode{S, Trivial}}, state_str::AbstractString) where{S}
    if state_str == "Emp"
        return [1, 0]
    elseif state_str == "Occ"
        return [0, 1]
    else
        error("Unkown state: $(state_str)")
    end
end


function _state(::Type{<:HardCoreBosonNode{S, <:AbstractIrrep}}, state_str::AbstractString) where{S}
    if state_str == "Emp"
        return 0
    elseif state_str == "Occ"
        return 1
    else
        error("Unkown state: $(state_str)")
    end
end
=#

function _state(::Type{<:HardCoreBosonNode{S, Trivial}}, state_str::AbstractString) where{S}
    if state_str == "Emp"
        return (1,1)
    elseif state_str == "Occ"
        return (2,1)
    else
        error("Unkown state: $(state_str)")
    end
end

function _state(::Type{<:HardCoreBosonNode{S, <:AbstractIrrep}}, state_str::AbstractString) where{S}
    if state_str == "Emp"
        return (0,1)
    elseif state_str == "Occ"
        return (1,1)
    else
        error("Unkown state: $(state_str)")
    end
end