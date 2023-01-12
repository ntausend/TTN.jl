struct ITensorNode{I} <: PhysicalNode{Index, I}
    s::Int
    desc::AbstractString
    hilbertspace::Index{I}
    function ITensorNode(pos, _idx::Index{I}) where{I}
        t_S = tags(_idx)
        #id_st = findfirst(t -> t == "Site", t_S)
        #isnothing(id_st) ||  error("Index given to the node $_idx is not derived from a ITensor hilbertspace.")
        (3 ≥ length(t_S)≥2 && string(t_S[1]) == "Site") || 
                        error("Index given to the node $_idx is not derived from a ITensor hilbertspace.")
        idx = length(t_S) == 3 ? _idx : addtags(_idx, "n=$pos")
        desc = string(t_S[2]) * " $pos"
        return new{I}(pos, desc, idx)
    end
end


#=
function TrivialNode_it(pos::Int, desc::AbstractString="", local_dim::Int = 2)
    idx = Index(local_dim, tags = desc)
    return ITensorNode(pos, idx)
end
=#

function ITensorNode(pos::Int, type::AbstractString; kwargs...)
    idx = addtags(siteind(type; kwargs...), "n=$pos")
    return ITensorNode(pos, idx)
end

index(nd::ITensorNode) = hilbertspace(nd)
state(nd::ITensorNode, st::AbstractString) = state(index(nd), st)
space(nd::ITensorNode) = space(index(nd))