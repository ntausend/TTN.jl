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

function state_dict(::Type{<:HardCoreBosonNode})
    return Dict{String, Vector{Int}}("Emp" => [1, 0], "Occ" => [0, 1])
end

function charge_dict(::Type{HardCoreBosonNode{S,I}}) where{S,I}
    if I == Trivial
        return nothing
    end
    return Dict{String, I}("Emp" => I(0), "Occ" => I(1))
end