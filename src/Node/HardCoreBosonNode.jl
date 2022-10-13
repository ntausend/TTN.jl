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

function spaces(::HardCoreBosonNode{S,I}) where {S,I}
    if I == Trivial
        return 2
    else
        [0 => 1, 1 => 1]
    end
end

state(::HardCoreBosonNode,::Val{:Occ}) = [0, 1]
state(::HardCoreBosonNode,::Val{:Emp}) = [1, 0]