abstract type AbstractProjTPO{N<:AbstractNetwork, T, B<:AbstractBackend} end

network(projTPO::AbstractProjTPO) = projTPO.net
tensor_product_operator(projTPO::AbstractProjTPO) = projTPO.tpo
ortho_center(projTPO::AbstractProjTPO) = Tuple(projTPO.ortho_center)

backend(::Type{<:AbstractProjTPO{N, T, B}}) where{N, T, B} = B
backend(projTPO::AbstractProjTPO) = backend(typeof(projTPO))


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