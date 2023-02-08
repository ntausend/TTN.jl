abstract type AbstractProjTPO{N<:AbstractNetwork, T, B<:AbstractBackend} end

network(projTPO::AbstractProjTPO) = projTPO.net
tensor_product_operator(projTPO::AbstractProjTPO) = projTPO.tpo
ortho_center(projTPO::AbstractProjTPO) = Tuple(projTPO.ortho_center)

backend(::Type{<:AbstractProjTPO{N, T, B}}) where{N, T, B} = B
backend(projTPO::AbstractProjTPO) = backend(typeof(projTPO))

function full_contraction(ttn::TreeTensorNetwork, tpo::AbstractTensorProductOperator)
    ptpo = ProjectedTensorProductOperator(ttn, tpo)
    return full_contraction(ttn, ptpo)
end
function full_contraction(ttn::TreeTensorNetwork, ptpo::AbstractProjTPO)
    # set the ptpo to the correct position of the ttn
    ptpo = set_position!(ptpo, ttn)
    oc = ortho_center(ttn)

    # get the action of the operator on the orthogonlity center
    action = ∂A(ptpo, oc)
    T = ttn[oc]
    # build the contraction
    return dot(T, action(T))
end

Base.getindex(projTPO::AbstractProjTPO, l::Int, p::Int) = getindex(projTPO, (l,p))
Base.getindex(projTPO::AbstractProjTPO, pos::Tuple{Int, Int}) = environments(projTPO, pos)

function set_position!(pTPO::AbstractProjTPO{N,T}, ttn::TreeTensorNetwork{N,T}) where {N,T}
    oc_projtpo = ortho_center(pTPO)
    oc_ttn     = ortho_center(ttn)
    # both structures should be gauged.. otherwise no real thing todo
    @assert !any(oc_ttn     .== -1)
    @assert !any(oc_projtpo .== -1)

    all(oc_projtpo .== oc_ttn) && return pTPO

    # path connecting both orthogonality centers
    pth = connecting_path(network(ttn), oc_projtpo, oc_ttn)

    if !isnothing(pth)
        pth = vcat(oc_projtpo, pth)
        for (jj,pk) in enumerate(pth[1:end-1])
            ism = ttn[pk]
            pTPO = update_environments!(pTPO, ism, pk, pth[jj+1])
        end
        pTPO.ortho_center .= oc_ttn
    end
    return pTPO
end