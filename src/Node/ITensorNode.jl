function _get_description(_tags)
    tags_n = string.(removetags(_tags, "Site"))
    id_n   = findfirst(t -> occursin("n=",t), tags_n)
    isnothing(id_n) || deleteat!(tags_n, id_n)
    return only(tags_n)
end


struct ITensorNode{I} <: PhysicalNode{Index, I}
    s::Int
    desc::AbstractString
    hilbertspace::Index{I}
    function ITensorNode(pos, _idx::Index{I}) where{I}
        t_S = tags(_idx)
        hastags(t_S, "Site") || throw(IndexMissmatchException(_idx, "Not derived from a ITensor hilbertspace"))
        desc = _get_description(t_S)
        idx = length(t_S) == 3 ? _idx : addtags(_idx, "n=$pos")
        #desc = desc# * " $pos"
        return new{I}(pos, desc, idx)
    end
end

function ITensorNode(pos::Int, type::AbstractString; kwargs...)
    idx = addtags(siteind(type; kwargs...), "n=$pos")
    return ITensorNode(pos, idx)
end

index(nd::ITensorNode) = hilbertspace(nd)
state(nd::ITensorNode, st::Union{AbstractString, Integer}) = state(index(nd), st)
space(nd::ITensorNode) = space(index(nd))

#struct ITensorNodeConverstionError <: Exception end
#Base.showerror(io::IO, ::ITensorNodeConverstionError) = print(io, "Tried to converted a Node to ITensor node which is not compatible.")
#ITensorNode(nd::TrivialNode{S,I}) where{S,I} = throw(ITensorNodeConverstionError())

#=
function TrivialNode_it(pos::Int, desc::AbstractString="", local_dim::Int = 2)
    idx = Index(local_dim, tags = desc)
    return ITensorNode(pos, idx)
end
=#
#=
# convert from Trivial node to ITensor node if requirements are there
ITensorNode(nd::TrivialNode{Index, Int64}) = ITensorNode(nd.s,nd.hilbertspace)

=#