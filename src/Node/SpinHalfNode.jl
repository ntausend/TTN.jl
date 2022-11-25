#=
TODO: Include Q_{U1} = \sum_j S_{z,j} and Q_{P} = \prod_j S_{z,j} conservation => basic copy of Hardcore boson node
=#
struct SpinHalfNode{S<:IndexSpace, I<:Sector} <: PhysicalNode{S,I}
    s::Int
    hilbertspace::S
    desc::AbstractString
    function SpinHalfNode(pos::Int, desc::AbstractString="";
                          conserve_qns::Bool = false,
                          conserve_parity::Bool = conserve_qns)
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
        return new{sp, sectortype(sp)}(pos, sp(sectors), "SH "*desc)
    end
end

function spaces(::SpinHalfNode{S,I}) where {S,I}
    if I == Trivial
        return 2
    else
        [0 => 1, 1 => 1]
    end
end


state(::SpinHalfNode,::Val{:Up}) = [1, 0]
state(::SpinHalfNode,::Val{:Down}) = [0, 1]
state(::SpinHalfNode,::Val{:Right}) = [1/sqrt(2), 1/sqrt(2)]
state(::SpinHalfNode,::Val{:Left}) = [1/sqrt(2), -1/sqrt(2)]


#=
function id(nd::SpinHalfNode)
    if sectortype(nd) == Trivial
        id = [1 0; 0 1]
        sp  = ℂ^2
        mpo = TensorMap(id, sp ← sp)
    else
        if sectortype(nd) == U1Irrep
            sp  = U1Space(0 => 1, 1 => 1)
            rep = Irrep[U₁]
        else
            sp  = Z2Space(0 => 1, 1 => 1)
            rep = Irrep[ℤ₂]
        end
        mpo = TensorMap(zeros, Float64, sp, sp)
        blocks(mpo)[rep(0)] .= [1]
        blocks(mpo)[rep(1)] .= [1]   
    end
    return mpo
end
function sig_z(nd::SpinHalfNode)
    if sectortype(nd) == Trivial
        σ_z = [1 0; 0 -1]
        sp  = ℂ^2
        mpo = TensorMap(σ_z, sp ← sp)
    else
        if sectortype(nd) == U1Irrep
            sp  = U1Space(0 => 1, 1 => 1)
            rep = Irrep[U₁]
        else
            sp  = Z2Space(0 => 1, 1 => 1)
            rep = Irrep[ℤ₂]
        end
        mpo = TensorMap(zeros, Float64, sp, sp)
        blocks(mpo)[rep(0)] .= [1]
        blocks(mpo)[rep(1)] .= [-1]   
    end
    return mpo
end

function sig_p(nd::SpinHalfNode)
    if sectortype(nd) == Trivial
        σ_z = [0 1; 0 0]
        sp  = ℂ^2
        mpo = TensorMap(σ_z, sp ← sp)
    else
        error("Not sure how to implement QN currrently, how to solve the non-trivial flow?")
    end
    return mpo
end

function sig_m(nd::SpinHalfNode)
    if sectortype(nd) == Trivial
        σ_z = [0 0; 1 0]
        sp  = ℂ^2
        mpo = TensorMap(σ_z, sp ← sp)
    else
        error("Not sure how to implement QN currrently, how to solve the non-trivial flow?")
    end
    return mpo
end


function sig_x(nd::SpinHalfNode)
    sectortype(nd) == Trivial || throw(QuantumNumberMissmatch())
    return sig_p(nd) + sig_m(nd)
end

function sig_y(nd::SpinHalfNode)
    sectortype(nd) == Trivial || throw(QuantumNumberMissmatch())
    return -im*(sig_p(nd) - sig_m(nd))
end
=#